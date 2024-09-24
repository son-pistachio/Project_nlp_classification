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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
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

# 데이터 분할
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=num_seed, stratify=df['label1'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=num_seed, stratify=temp_df['label1'])

CHECKPOINT_NAME = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'  # SBERT 모델

# 데이터셋 클래스 수정
class SBERTDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']
        return text, label

# 데이터셋 생성
train_data = SBERTDataset(train_df)
val_data = SBERTDataset(val_df)
test_data = SBERTDataset(test_df)

# collate_fn 정의
def collate_batch(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels)
    return list(texts), labels

num_workers = multiprocessing.cpu_count() // 2
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)

# Lightning 모듈 수정
class SBERTLightningModel(LightningModule):
    def __init__(self, sbert_model_name, num_labels=num_classes, lr=lr, total_steps=None):
        super(SBERTLightningModel, self).__init__()
        self.save_hyperparameters()
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.sbert_model.to(self.device)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.total_steps = total_steps

    def forward(self, texts):
        embeddings = self.sbert_model.encode(texts, convert_to_tensor=True, device=self.device)
        logits = self.fc(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        labels = labels.to(self.device)
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        labels = labels.to(self.device)
        outputs = self(texts)
        loss = self.loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        self.val_predictions.append(preds)
        self.val_targets.append(labels)
        self.val_losses.append(loss)
        self.log('val_loss_step', loss, prog_bar=True)

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
        texts, labels = batch
        labels = labels.to(self.device)
        outputs = self(texts)
        _, preds = torch.max(outputs, dim=1)
        self.test_predictions.append(preds)
        self.test_targets.append(labels)

    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_targets = []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)

        # 분산 환경에서 모든 프로세스의 결과를 모음
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
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

# WandbLogger 설정
now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
wandb_logger = WandbLogger(project="bert-classification", log_model=True, name="sbert"+now_sys)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/sbert/',
    filename='sbert-{epoch:02d}-{val_loss:.2f}',
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
bert_model = SBERTLightningModel(CHECKPOINT_NAME, total_steps=total_steps)

trainer.fit(bert_model, train_loader, val_loader)
trainer.test(bert_model, test_loader)

wandb.finish()
