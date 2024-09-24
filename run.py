import os
import subprocess

# 모델 학습 스크립트를 리스트로 정의
model_scripts = ['bert.py','distilBERT.py','electra.py','roberta.py','sbert.py', 'gpt.py']

def run_model(script_name):
    subprocess.run(['python', script_name])

for script in model_scripts:
    print(f"Running {script}...")
    run_model(script)
    print(f"Finished {script}\n")
