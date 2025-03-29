#!/bin/bash

# 고정 디렉토리로 이동 (CodeDeploy 경로 말고!)
cd /home/ubuntu/langchain-svc || exit 1

# venv 삭제 후 새로 만들기 (매번 덮어쓰기 가능)
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
