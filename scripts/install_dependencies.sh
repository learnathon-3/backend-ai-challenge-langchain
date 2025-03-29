#!/bin/bash

# 현재 스크립트 기준 상위 디렉토리 (프로젝트 루트)
DEPLOY_DIR=$(dirname "$(readlink -f "$0")")/..

# 해당 경로로 이동
cd "$DEPLOY_DIR"

# 가상환경 생성 및 패키지 설치
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
