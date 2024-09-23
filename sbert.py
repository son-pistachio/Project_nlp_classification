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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer  # For SBERT

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

CHECKPOINT_NAME = 'sentence-transformers/bert-base-nli-mean-tokens'  # SBERT model

class SBERTDataset(Dataset):
    def __init__(self, dataframe, sbert_model):
        self.data = dataframe
        self.sbert_model = SentenceTransformer(sbert_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']
        # Use SBERT to encode the text directly into embeddings
        embeddings = self.sbert_model.encode(text, convert_to_tensor=True)
        return {'embeddings': embeddings}, torch.tensor(label)

# train, test 데이터셋 생성
train_data = SBERTDataset(train_df, CHECKPOINT_NAME)
val_data = SBERTDataset(val_df, CHECKPOINT_NAME)
test_data = SBERTDataset(test_df, CHECKPOINT_NAME)

train_loader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

# Lightning 모듈 정의
class SBERTLightningModel(LightningModule):
    def __init__(self, sbert_model_name, num_labels=12, lr=1e-5):
        super(SBERTLightningModel, self).__init__()
        self.save_hyperparameters()
        # SBERT returns embeddings, so we don't need a full transformer model here
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.fc = nn.Linear(768, num_labels)  # SBERT's embedding size is 768
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = inputs['embeddings']
        outputs = self(embeddings)
        loss = self.loss_fn(outputs, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        _, preds = torch.max(outputs, dim=1)
        self.val_predictions.append(preds)
        self.val_targets.append(labels)
        self.val_losses.append(self.loss_fn(outputs, labels))

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_targets = []
        self.val_losses = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        loss = torch.mean(torch.stack(self.val_losses))

        # 분산 환경에서 모든 프로세스의 결과를 모음
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

        # 분산 환경에서 모든 프로세스의 결과를 모음
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
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

# WandbLogger 설정
wandb.init(project="sbert-classification")
wandb_logger = WandbLogger(project="sbert-classification", log_model=True)

sbert_model = SBERTLightningModel(CHECKPOINT_NAME)
trainer = Trainer(
    max_epochs=10,
    deterministic=True,
    logger=wandb_logger
)

trainer.fit(sbert_model, train_loader, val_loader)
trainer.test(sbert_model, test_loader)

wandb.finish()
