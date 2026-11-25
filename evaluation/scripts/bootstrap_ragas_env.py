"""可交接的固定环境安装任务；不读取.env、不调用模型、不修改项目依赖。"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'evaluation/.ragas_env'
REPORT=ROOT/'evaluation/artifacts/answer_quality_environment'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=REPORT/'bootstrap.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        executable=TARGET/'Scripts/python.exe'
        env=dict(os.environ,RAGAS_DO_NOT_TRACK='true',PYTHONIOENCODING='utf-8')
        with (REPORT/'installation.log').open('w',encoding='utf-8') as log:
            if not executable.exists():
                result=subprocess.run([sys._base_executable,'-m','venv',str(TARGET)],stdout=log,stderr=subprocess.STDOUT,env=env)
                if result.returncode:raise RuntimeError('Venv creation failed; return installation.log')
            packages=['ragas==0.4.4','dashscope==1.27.8','langchain-community==0.4.2','python-dotenv==1.2.4']
            result=subprocess.run([str(executable),'-m','pip','install','--disable-pip-version-check','--no-input',
                                   '--index-url','https://pypi.org/simple',*packages],stdout=log,stderr=subprocess.STDOUT,env=env)
            if result.returncode:raise RuntimeError('Installation failed; do not change pins, return installation.log')
        result=subprocess.run([str(executable),'-m','pip','check'],capture_output=True,text=True,encoding='utf-8',env=env)
        (REPORT/'pip_check.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Dependency check failed')
        result=subprocess.run([str(executable),'-m','pip','freeze'],capture_output=True,text=True,encoding='utf-8',env=env,check=True)
        (REPORT/'requirements.freeze.txt').write_text(result.stdout,encoding='utf-8')
        code="import importlib.metadata,json; from ragas.metrics import ResponseRelevancy,Faithfulness; print(json.dumps({'ragas':importlib.metadata.version('ragas'),'response_relevancy_import':True,'faithfulness_import':True}))"
        result=subprocess.run([str(executable),'-c',code],capture_output=True,text=True,encoding='utf-8',env=env)
        (REPORT/'import_check.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Metric import check failed')
        report={'status':'ready','executable':str(executable),'ragas_version':'0.4.4','project_main_env_modified':False,
                'env_file_read':False,'paid_model_calls':0,'pip_check':'passed','metric_imports':'passed'}
        (REPORT/'ready.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
