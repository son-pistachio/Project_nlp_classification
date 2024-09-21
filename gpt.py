import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

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

df = pd.read_csv("/home/son/ml/hanyang/datasets/final_data.csv")

# train, val(15%), test(15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=num_seed, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=num_seed, stratify=temp_df['label1'])

CHECKPOINT_NAME = 'gpt2'  # Change to GPT-2 checkpoint

class GPT2TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe        
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_pretrained)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 doesn't have a padding token, so use eos_token
  
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
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, torch.tensor(label)

# train, test 데이터셋 생성
train_data = GPT2TokenDataset(train_df, CHECKPOINT_NAME)
val_data = GPT2TokenDataset(val_df, CHECKPOINT_NAME)
test_data = GPT2TokenDataset(test_df, CHECKPOINT_NAME)

train_loader = DataLoader(train_data, batch_size=8, shuffle=False, num_workers=4)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

# Lightning 모듈 정의
class GPT2LightningModel(LightningModule):
    def __init__(self, gpt2_pretrained, num_labels=12, lr=1e-5):
        super(GPT2LightningModel, self).__init__()
        self.save_hyperparameters()
        self.gpt2 = GPT2Model.from_pretrained(gpt2_pretrained)
        self.fc = nn.Linear(768, num_labels)  # GPT-2's hidden size is 768
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        cls_output = last_hidden_state[:, -1, :]  # Use the last token's output
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
wandb.init(project="gpt-classification")
wandb_logger = WandbLogger(project="gpt-classification", log_model=True)

gpt2_model = GPT2LightningModel(CHECKPOINT_NAME)
trainer = Trainer(
    max_epochs=2,
    deterministic=True,
    logger=wandb_logger
)

trainer.fit(gpt2_model, train_loader, val_loader)
trainer.test(gpt2_model, test_loader)

wandb.finish()
