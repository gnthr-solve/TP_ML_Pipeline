@echo off

for /f "tokens=2 delims=:" %%a in ('findstr /c:"name:" environment.yml') do set ML_Pipeline=%%a
python -m venv %ML_Pipeline%
%ML_Pipeline%\Scripts\activate
pip install -r requirements.txt
