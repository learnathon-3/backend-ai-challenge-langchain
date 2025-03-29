#!/bin/bash
cd /home/ubuntu/langchain-svc
source venv/bin/activate
nohup gunicorn -c gunicorn_conf.py app.main:app &
