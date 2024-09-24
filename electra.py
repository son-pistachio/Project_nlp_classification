import multiprocessing
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import datetime
import yaml
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import ElectraModel, ElectraTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint

if torch.__version__ >= '2.0':
    torch.set_float32_matmul_precision('high')

num_seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

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

set_seed(num_seed)

# YAML 파일에서 설정 읽기
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        
        lr = config['lr']
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']
        max_len = config['max_len']
        num_classes = config['num_classes']
    return float(lr), batch_size, max_epochs, max_len, num_classes

df = pd.read_csv("/home/son/ml/hanyang/datasets/final_data.csv")
lr, batch_size, max_epochs, max_len, num_classes = load_config('config.yaml')

# train, val(15%), test(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=num_seed, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=num_seed, stratify=temp_df['label1'])

CHECKPOINT_NAME = 'monologg/koelectra-base-v3-discriminator'

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe        
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer_pretrained)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']
        tokens = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len, add_special_tokens=True
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, torch.tensor(label)

# train, test 데이터셋 생성
train_data = TokenDataset(train_df, CHECKPOINT_NAME)
val_data = TokenDataset(val_df, CHECKPOINT_NAME)
test_data = TokenDataset(test_df, CHECKPOINT_NAME)

num_workers = multiprocessing.cpu_count() // 2
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class ELECTRALightningModel(LightningModule):
    def __init__(self, bert_pretrained, num_labels=num_classes, lr=lr, total_steps=None):
        super(ELECTRALightningModel, self).__init__()
        self.save_hyperparameters()
        self.bert = ElectraModel.from_pretrained(bert_pretrained)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.total_steps = total_steps

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
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
        self.val_predictions.append(preds)
        self.val_targets.append(labels)
        self.val_losses.append(loss)
        # self.log('val_loss_step', loss, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_targets = []
        self.val_losses = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        loss = torch.mean(torch.stack(self.val_losses))

        # 분산 환경에서 모든 프로세스의 결과를 모음
        if self.trainer.world_size > 1:
            preds = self.all_gather(preds)
            targets = self.all_gather(targets)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        _, preds = torch.max(outputs, dim=1)
        self.test_predictions.append(preds)
        self.test_targets.append(labels)

    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_targets = []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)

        if self.trainer.world_size > 1:
            preds = self.all_gather(preds)
            targets = self.all_gather(targets)

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')

        self.log('test_acc', acc)
        self.log('test_f1', f1)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, correct_bias=False, no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

# WandbLogger 설정
now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
wandb_logger = WandbLogger(project="bert-classification", log_model=True, name="electra_"+now_sys)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/electra/',
    filename='electra-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    max_epochs=max_epochs,
    deterministic=True,
    logger=wandb_logger,
    callbacks=[checkpoint_callback]

)
total_steps = len(train_loader) * trainer.max_epochs
bert_model = ELECTRALightningModel(CHECKPOINT_NAME, total_steps=total_steps)

trainer.fit(bert_model, train_loader, val_loader)
trainer.test(bert_model, test_loader)

wandb.finish()