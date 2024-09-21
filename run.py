import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

torch.set_float32_matmul_precision('high')

num_seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

seed_everything(num_seed, workers=True)

df = pd.read_csv("/home/son/ml/hanyang/datasets/final_data.csv")

# train, val(15%), test(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=num_seed, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=num_seed, stratify=temp_df['label1'])

CHECKPOINT_NAME = 'kykim/bert-kor-base'

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe        
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
  
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
        token_type_ids = torch.zeros_like(attention_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, torch.tensor(label)

# train, test 데이터셋 생성
train_data = TokenDataset(train_df, CHECKPOINT_NAME)
val_data = TokenDataset(val_df, CHECKPOINT_NAME)
test_data = TokenDataset(test_df, CHECKPOINT_NAME)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

# Lightning 모듈 정의
class BertLightningModel(LightningModule):
    def __init__(self, bert_pretrained, num_labels=12, lr=1e-5):
        super(BertLightningModel, self).__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
        acc = torch.sum(preds == labels).item() / len(labels)
        
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

        
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

bert_model = BertLightningModel(CHECKPOINT_NAME)
trainer = Trainer(max_epochs=2)
trainer.fit(bert_model, train_loader, val_loader)
trainer.test(bert_model, test_loader)

