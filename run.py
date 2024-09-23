import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, DistilBertModel, RobertaModel, ElectraModel, AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

torch.set_float32_matmul_precision('high')

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

# 데이터 로드 및 분리
df = pd.read_csv("/home/son/ml/hanyang/datasets/final_data.csv")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label1'])

# Dataset 클래스
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained, use_token_type_ids=True):
        self.data = dataframe        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)
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

        # token_type_ids는 필요할 때만 반환
        if self.use_token_type_ids:
            token_type_ids = tokens['token_type_ids'].squeeze(0)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, torch.tensor(label)
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}, torch.tensor(label)

# DataLoader 생성 함수
def create_dataloaders(train_df, val_df, test_df, tokenizer_name, use_token_type_ids, batch_size=4):
    train_data = TokenDataset(train_df, tokenizer_name, use_token_type_ids=use_token_type_ids)
    val_data = TokenDataset(val_df, tokenizer_name, use_token_type_ids=use_token_type_ids)
    test_data = TokenDataset(test_df, tokenizer_name, use_token_type_ids=use_token_type_ids)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# 공통 모델 클래스
class BaseLightningModel(LightningModule):
    def __init__(self, model, num_labels=12, lr=1e-5, dropout_prob=0.3):
        super(BaseLightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
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

# 모델 및 데이터셋 설정
models = {
    "BERT": ("kykim/bert-kor-base", BertModel, True),
    "DistilBERT": ("monologg/distilkobert", DistilBertModel, False),
    "RoBERTa": ("klue/roberta-base", RobertaModel, False),
    "Electra": ("monologg/koelectra-base-v3-discriminator", ElectraModel, False),
}

# 모델 학습 및 평가 함수
def train_and_test_model(model_name, model_class, tokenizer_name, use_token_type_ids, train_loader, val_loader, test_loader, num_epochs=1):
    model = BaseLightningModel(AutoModel.from_pretrained(tokenizer_name))
    trainer = Trainer(max_epochs=num_epochs, deterministic=True)
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    return result

# 각 모델 학습 및 평가 실행
results = {}
for model_name, (tokenizer_name, model_class, use_token_type_ids) in models.items():
    print(f"Training and evaluating {model_name}...")
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, tokenizer_name, use_token_type_ids)
    results[model_name] = train_and_test_model(model_name, model_class, tokenizer_name, use_token_type_ids, train_loader, val_loader, test_loader)

# 결과 출력
for model_name, result in results.items():
    print(f"{model_name} Model Test Results:", result)
