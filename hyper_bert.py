# bert_optuna_final.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import multiprocessing
import wandb
import datetime
import yaml
import optuna

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


if torch.__version__ >= '2.0':
    torch.set_float32_matmul_precision('high')

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터 설정 로드
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        # 각 하이퍼파라미터를 적절한 형으로 변환
        config['lr'] = [float(x) for x in config['lr']]
        config['dropout_rate'] = [float(x) for x in config['dropout_rate']]
        config['weight_decay'] = [float(x) for x in config['weight_decay']]
        config['warmup_steps'] = [int(x) for x in config['warmup_steps']]
        config['max_len'] = [int(x) for x in config['max_len']]
        return config

config = load_config('hyper_config.yaml')

# 텍스트 전처리 함수
def clean_text(text):
    text = text.strip()
    text = ' '.join(text.split())
    return text

# 데이터 로드 및 전처리
df = pd.read_csv('/home/son/ml/nlp_classification/datasets/final_data.csv')
df['text'] = df['text'].apply(clean_text)

# 데이터 분할 (train_val_df and test_df)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label1'])
train_val_df = train_val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 모델 및 토크나이저 체크포인트
CHECKPOINT_NAME = 'kykim/bert-kor-base'


# 데이터셋 클래스 정의
class TokenDataset(Dataset):
    def __init__(self, dataframe, tokenizer_pretrained, max_len, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)
        self.max_len = max_len
        self.augment = augment
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label1']

        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            add_special_tokens=True
        )

        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        token_type_ids = tokens['token_type_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }, torch.tensor(label)

# 데이터로더 설정
num_workers = multiprocessing.cpu_count() // 2

# 모델 정의
class BertLightningModel(LightningModule):
    def __init__(self, bert_pretrained, num_labels, lr, dropout_rate, weight_decay, warmup_steps, optimizer_name, class_weights):
        super(BertLightningModel, self).__init__()
        self.save_hyperparameters(ignore=['class_weights'])
        self.bert = BertModel.from_pretrained(bert_pretrained)
        self.dropout = nn.Dropout(dropout_rate)
        # 분류기 헤드 개선
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
        # 클래스 가중치 적용
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        loss = self.loss_fn(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
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

        preds = preds.cpu()
        targets = targets.cpu()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs)
        preds = torch.argmax(outputs, dim=1)
        self.test_predictions.append(preds)
        self.test_targets.append(labels)

    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_targets = []

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)

        preds = preds.cpu()
        targets = targets.cpu()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')

        self.log('test_acc', acc)
        self.log('test_f1', f1)

        # 혼동 행렬 및 분류 보고서 생성
        report = classification_report(targets, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv', index=True)

        cm = confusion_matrix(targets, preds)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv('confusion_matrix.csv', index=False)

        # 테스트 결과 저장
        results_df = pd.DataFrame({
            'true_label': targets,
            'predicted_label': preds
        })
        results_df.to_csv('test_results.csv', index=False)

    def configure_optimizers(self):
        optimizer = None
        if self.hparams.optimizer_name == 'AdamW':
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay
            )
        # elif self.hparams.optimizer_name == 'Adam':
        #     optimizer = Adam(
        #         self.parameters(),
        #         lr=self.hparams.lr,
        #         weight_decay=self.hparams.weight_decay
        #     )
        
            # 총 학습 스텝 수 계산

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

# 클래스 가중치 계산 함수
def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return torch.tensor(class_weights, dtype=torch.float)

# Optuna 목적 함수 정의
def objective(trial):
    # 하이퍼파라미터 샘플링
    lr = trial.suggest_float('lr', config['lr'][0], config['lr'][1], log=True)
    batch_size = trial.suggest_categorical('batch_size', config['batch_size'])
    dropout_rate = trial.suggest_float('dropout_rate', config['dropout_rate'][0], config['dropout_rate'][1])
    weight_decay = trial.suggest_float('weight_decay', config['weight_decay'][0], config['weight_decay'][1], log=True)
    warmup_steps = trial.suggest_int('warmup_steps', config['warmup_steps'][0], config['warmup_steps'][1])
    max_len = trial.suggest_int('max_len', config['max_len'][0], config['max_len'][1])
    optimizer_name = trial.suggest_categorical('optimizer_name', config['optimizer_name'])
    num_classes = config['num_classes']

    # Stratified K-Fold 교차 검증 설정
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_f1_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(train_val_df, train_val_df['label1'])):
        fold_train_df = train_val_df.iloc[train_index].reset_index(drop=True)
        fold_val_df = train_val_df.iloc[val_index].reset_index(drop=True)

        # 클래스 가중치 계산 (각 폴드의 훈련 데이터로)
        fold_labels = fold_train_df['label1'].values
        fold_class_weights = compute_class_weights(fold_labels)

        # 데이터셋 생성
        train_data = TokenDataset(fold_train_df, CHECKPOINT_NAME, max_len, augment=True)
        val_data = TokenDataset(fold_val_df, CHECKPOINT_NAME, max_len)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 모델 정의
        model = BertLightningModel(
            bert_pretrained=CHECKPOINT_NAME,
            num_labels=num_classes,
            lr=lr,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            optimizer_name=optimizer_name,
            class_weights=fold_class_weights
        )

        # 콜백 설정
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=False,
            mode='min',
            min_delta=0.0001
        )

        # 로거 설정
        now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
        wandb_logger = WandbLogger(project="bert-classification-hyperparameter", log_model=False, name="bert_"+now_sys)

        # 트레이너 설정
        trainer = Trainer(
            precision=16,
            max_epochs=100,
            deterministic=True,
            logger=wandb_logger,
            callbacks=[early_stop_callback],
            enable_checkpointing=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else None,
        )

        # 학습
        trainer.fit(model, train_loader, val_loader)

        # 검증 F1 스코어 저장
        val_f1 = trainer.callback_metrics.get("val_f1")
        if val_f1 is not None:
            val_f1_scores.append(val_f1.item())
        else:
            val_f1_scores.append(0.0)

        # W&B 세션 종료
        wandb.finish()

    # K-Fold 검증 점수의 평균 계산
    mean_val_f1 = np.mean(val_f1_scores)
    return mean_val_f1

if __name__ == '__main__':
    
    # 시드 고정
    def set_seed(seed: int = 42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_everything(seed, workers=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed()
    # Optuna 스터디 생성
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Val F1: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 최적의 하이퍼파라미터로 모델 학습 및 평가
    best_params = trial.params

    lr = best_params['lr']
    batch_size = best_params['batch_size']
    dropout_rate = best_params['dropout_rate']
    weight_decay = best_params['weight_decay']
    warmup_steps = best_params['warmup_steps']
    max_len = best_params['max_len']
    optimizer_name = best_params['optimizer_name']
    num_classes = config['num_classes']

    # train_val_df를 train_df와 val_df로 분할
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['label1'])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 클래스 가중치 계산 (train_df 사용)
    labels = train_df['label1'].values
    class_weights = compute_class_weights(labels)

    # 데이터셋 생성
    train_data = TokenDataset(train_df, CHECKPOINT_NAME, max_len, augment=True)
    val_data = TokenDataset(val_df, CHECKPOINT_NAME, max_len)
    test_data = TokenDataset(test_df, CHECKPOINT_NAME, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 모델 정의
    model = BertLightningModel(
        bert_pretrained=CHECKPOINT_NAME,
        num_labels=num_classes,
        lr=lr,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        optimizer_name=optimizer_name,
        class_weights=class_weights
    )

    # 콜백 설정
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min',
        min_delta=0.001
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/bert/',
        filename='bert-best-{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode='min',
    )

    # 로거 설정
    now_sys = datetime.datetime.now().strftime("%m%d_%H%M")
    wandb_logger = WandbLogger(project="bert-classification-final", log_model=True, name="bert_"+now_sys)

    trainer = Trainer(
        precision=16,
        max_epochs=100,
        deterministic=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    # 모델 학습
    trainer.fit(model, train_loader, val_loader)

    # 테스트 데이터로 평가
    trainer.test(model, test_loader)

    # 모델 저장
    trainer.save_checkpoint("checkpoints/bert/final_model.ckpt")

    wandb.finish()
