@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001
call conda activate rag
python -u scripts\test_fast.py
