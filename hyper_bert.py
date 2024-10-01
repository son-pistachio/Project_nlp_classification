# bert_optuna.py
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
import optuna
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
        # lr과 dropout_rate를 float로 변환
        config['lr'] = [float(x) for x in config['lr']]
        config['dropout_rate'] = [float(x) for x in config['dropout_rate']]
        return config

config = load_config('hyper_config.yaml')

df = pd.read_csv("/home/son/ml/nlp_classification/datasets/final_data.csv")

# 데이터 분할 및 저장
if not os.path.exists('./data/train.csv'):
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=num_seed, stratify=df['label1'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=num_seed, stratify=temp_df['label1'])
    os.makedirs('./data', exist_ok=True)
    train_df.to_csv('./data/train.csv', index=False)
    val_df.to_csv('./data/val.csv', index=False)
    test_df.to_csv('./data/test.csv', index=False)
else:
    # 저장된 CSV 파일 불러오기
    train_df = pd.read_csv('./data/train.csv')
    val_df = pd.read_csv('./data/val.csv')
    test_df = pd.read_csv('./data/test.csv')

CHECKPOINT_NAME = 'kykim/bert-kor-base'

class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained, max_len):
        self.data = dataframe        
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.max_len = max_len
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']
        
        tokens = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_len, add_special_tokens=True
        )

        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        token_type_ids = tokens['token_type_ids'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, torch.tensor(label)

num_workers = multiprocessing.cpu_count() // 2

class BertLightningModel(LightningModule):
    def __init__(self, bert_pretrained, num_labels, lr, dropout_rate):
        super(BertLightningModel, self).__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = output.last_hidden_state[:, 0, :]
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
        self.val_predictions.append(preds)
        self.val_targets.append(labels)
        self.val_losses.append(loss)

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_targets = []
        self.val_losses = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        loss = torch.mean(torch.stack(self.val_losses))

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

        # CSV로 저장
        results_df = pd.DataFrame({'true_label': targets, 'predicted_label': preds})
        results_df.to_csv('test_results.csv', index=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, correct_bias=False, no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

def objective(trial):
    # 하이퍼파라미터 범위를 config에서 가져옴
    lr = trial.suggest_float('lr', min(config['lr']), max(config['lr']), log=True)
    batch_size = trial.suggest_categorical('batch_size', config['batch_size'])
    dropout_rate = trial.suggest_float('dropout_rate', min(config['dropout_rate']), max(config['dropout_rate']))

    max_epochs = config['max_epochs']
    max_len = config['max_len']
    num_classes = config['num_classes']

    # 데이터셋 생성
    train_data = TokenDataset(train_df, CHECKPOINT_NAME, max_len)
    val_data = TokenDataset(val_df, CHECKPOINT_NAME, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 모델 정의
    model = BertLightningModel(
        bert_pretrained=CHECKPOINT_NAME,
        num_labels=num_classes,
        lr=lr,
        dropout_rate=dropout_rate
    )

    # 콜백 설정
    early_stop_callback = EarlyStopping(
        monitor='val_f1',  # 성능 지표를 val_f1으로 변경
        patience=3,
        verbose=True,
        mode='max'  # val_f1이 최대화되도록 설정
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',  # 성능 지표를 val_f1으로 변경
        dirpath='checkpoints/bert/',
        filename='bert-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode='max',
    )

    # 로거 설정
    now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_logger = WandbLogger(project="bert-classification-hyperparameter", log_model=True, name=f"bert_{now_sys}_{trial.number}")

    # 트레이너 설정
    trainer = Trainer(
        max_epochs=max_epochs,
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
    )

    # 학습
    trainer.fit(model, train_loader, val_loader)

    # 검증 F1 스코어 반환
    val_f1 = trainer.callback_metrics["val_f1"].item()

    # W&B 세션 종료
    wandb.finish()

    return val_f1

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')  # 방향을 maximize로 변경
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Val F1: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 최적의 하이퍼파라미터로 최종 모델 학습
    best_params = trial.params

    lr = best_params['lr']
    batch_size = best_params['batch_size']
    dropout_rate = best_params['dropout_rate']

    max_epochs = config['max_epochs']
    max_len = config['max_len']
    num_classes = config['num_classes']

    # 전체 데이터로 모델 재학습
    train_data = TokenDataset(train_df, CHECKPOINT_NAME, max_len)
    val_data = TokenDataset(val_df, CHECKPOINT_NAME, max_len)
    test_data = TokenDataset(test_df, CHECKPOINT_NAME, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = BertLightningModel(
        bert_pretrained=CHECKPOINT_NAME,
        num_labels=num_classes,
        lr=lr,
        dropout_rate=dropout_rate
    )

    # 콜백 설정
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=5,
        verbose=True,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath='checkpoints/bert/',
        filename='bert-best-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode='max',
    )

    # 로거 설정
    now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_logger = WandbLogger(project="bert-classification-hyperparameter", log_model=True, name=f"bert_final_{now_sys}")

    trainer = Trainer(
        max_epochs=max_epochs,
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
    )

    # 학습
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # 모델 저장
    trainer.save_checkpoint("checkpoints/bert/final_model.ckpt")

    wandb.finish()
