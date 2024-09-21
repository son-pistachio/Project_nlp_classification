import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, RobertaModel, ElectraModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

torch.set_float32_matmul_precision('high')

# YAML 파일에서 설정 로드
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

# Seed 고정 함수
def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed, workers=True)

set_seed(42)

# 데이터 로드
df = pd.read_csv("/home/son/ml/hanyang/datasets/final_data.csv")

# 데이터셋 분리: train, val(15%), test(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label1'])

# 모델 종류에 따라 다른 Tokenizer와 Model 선택
model_name = config['model']['name']
if model_name == "BERT":
    CHECKPOINT_NAME = "kykim/bert-kor-base"
    ModelClass = BertModel
    use_token_type_ids = True
elif model_name == "RoBERTa":
    CHECKPOINT_NAME = "klue/roberta-base"
    ModelClass = RobertaModel
    use_token_type_ids = False  # RoBERTa는 token_type_ids 사용 안함
elif model_name == "Electra":
    CHECKPOINT_NAME = "monologg/koelectra-base-v3-discriminator"
    ModelClass = ElectraModel
    use_token_type_ids = False  # ELECTRA도 token_type_ids 사용 안함

# TokenDataset 정의
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained, use_token_type_ids=True):
        self.data = dataframe        
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.use_token_type_ids = use_token_type_ids
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']
        tokens = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding='max_length', add_special_tokens=True
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        if self.use_token_type_ids:
            token_type_ids = tokens['token_type_ids'].squeeze(0)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, torch.tensor(label)
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}, torch.tensor(label)

# DataLoader 생성
train_data = TokenDataset(train_df, CHECKPOINT_NAME, use_token_type_ids=use_token_type_ids)
val_data = TokenDataset(val_df, CHECKPOINT_NAME, use_token_type_ids=use_token_type_ids)
test_data = TokenDataset(test_df, CHECKPOINT_NAME, use_token_type_ids=use_token_type_ids)

batch_size = config['train']['batch_size']
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# Lightning 모듈 정의
class BaseLightningModel(LightningModule):
    def __init__(self, model_class, bert_pretrained, num_labels, lr):
        super(BaseLightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = model_class.from_pretrained(bert_pretrained)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        logits = self.fc(cls_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

# WandbLogger 설정
wandb.init(project=config['wandb']['project_name'])
wandb_logger = WandbLogger(project=config['wandb']['project_name'], log_model=config['wandb']['log_model'])

bert_model = BaseLightningModel(
    model_class=ModelClass,
    bert_pretrained=CHECKPOINT_NAME,
    num_labels=config['model']['num_labels'],
    lr=float(config['train']['learning_rate'])
)

trainer = Trainer(
    max_epochs=config['train']['num_epochs'],
    deterministic=True,
    logger=wandb_logger
)

trainer.fit(bert_model, train_loader, val_loader)
trainer.test(bert_model, test_loader)

wandb.finish()
