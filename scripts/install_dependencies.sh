#!/bin/bash
cd /home/ubuntu/langchain-svc
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
