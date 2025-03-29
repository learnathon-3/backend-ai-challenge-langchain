#!/bin/bash
cd /home/ubuntu/langchain-svc
source venv/bin/activate
nohup gunicorn src.main:app &
