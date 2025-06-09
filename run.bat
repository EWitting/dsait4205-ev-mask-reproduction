@echo off

set PYTHONPATH=%cd%;%PYTHONPATH%
cd src
uv run main.py