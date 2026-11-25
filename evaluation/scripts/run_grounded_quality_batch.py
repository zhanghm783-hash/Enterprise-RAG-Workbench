"""一次执行证据约束回答＋同口径前后忠实度评分，保留旧实验。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'evaluation/artifacts/answer_quality_v4_extractive'
NEW_BATCH='evaluation/artifacts/answer_quality_v4_extractive'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def summarize(folder,scheme):
    manifest=read(folder/'manifest.json')
    if manifest['status'] not in ('complete','complete_with_undefined_or_errors'):raise ValueError('Scoring not finished')
    rows=[json.loads(s) for s in (folder/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len({r['question_id'] for r in rows})==24
    assert all(r['scheme']==scheme for r in rows)
    values=[r['score'] for r in rows if r['score'] is not None]
    saved=read(folder/'summary.json')[scheme]
    mean=sum(values)/len(values) if values else None
    assert mean==saved['mean_on_finite']
    assert manifest['prompt_language']=='zh'
    return {'questions':24,'mean':mean,'valid':len(values),'errors':sum(r['status']=='error' for r in rows),
            'undefined':sum(r['status']=='undefined' for r in rows),'missing_zero_sensitivity':sum(values)/24,
            'judge_input_tokens':manifest['judge_input_tokens'],'judge_output_tokens':manifest['judge_output_tokens'],
            'scores_sha256':hashlib.sha256((folder/'scores.jsonl').read_bytes()).hexdigest()},rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('This batch uses paid APIs; run with --execute, estimate 3-8 RMB')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    task_lock=OUTPUT/'batch.lock';fd=os.open(task_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    env=dict(os.environ,PYTHONIOENCODING='utf-8',RAGAS_DO_NOT_TRACK='true')
    backend=ROOT/'backend/.venv/Scripts/python.exe';ragas=ROOT/'evaluation/.ragas_env/Scripts/python.exe'
    scripts=ROOT/'evaluation/scripts'
    def run(name,command):
        print(name+': running',flush=True)
        with (OUTPUT/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([str(a) for a in command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' failed; preserve log and return, do not change versions or retry automatically')
        print(name+': finished',flush=True)
    try:
        raw_manifest=ROOT/'evaluation/artifacts/answer_quality_v3_grounded_max/manifest.json'
        if not raw_manifest.exists() or read(raw_manifest)['status']=='prepared':
            run('generate',[backend,scripts/'generate_grounded_answers.py','--model','qwen3.7-max','--allow-paid'])
        run('validate_quotes',[backend,scripts/'revalidate_grounded_quotes.py'])
        if not (OUTPUT/'ready.json').exists():
            run('extractive_answers',[backend,scripts/'build_extractive_answers.py'])
        old=ROOT/'evaluation/artifacts/answer_quality_v1/faithfulness_qwen37max_v1_zh2_vector'
        new=OUTPUT/'faithfulness_qwen37max_v1_zh2'
        def evaluate(name,batch,filter_name,folder):
            if (folder/'summary.json').exists() and read(folder/'manifest.json')['status'].startswith('complete'):
                print(name+': using saved complete results; no extra API calls',flush=True);return
            command=[ragas,scripts/'score_answer_quality_ragas.py','--batch-dir',batch,'--metric','faithfulness',
                     '--language','zh','--workers','4','--allow-paid']
            if filter_name:command.extend(['--schemes-filter',filter_name])
            if (folder/'manifest.json').exists():command.append('--resume')
            run(name,command)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(evaluate,'baseline_faithfulness','evaluation/artifacts/answer_quality_v1','vector',old),
                     executor.submit(evaluate,'improved_faithfulness',NEW_BATCH,'',new)]
            for future in futures:future.result()
        baseline_rows=[json.loads(s) for s in (old/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in baseline_rows:
            if row['status']=='ok':continue
            if row.get('error_type')!='LLMDidNotFinishException':raise ValueError('Unexpected baseline error; return logs')
            qid=row['question_id']
            repair=old.parent/('faithfulness_qwen37max_v1_zh2_vector_ids'+qid+'_max8192')
            if not (repair/'summary.json').exists():
                run('repair_'+qid,[ragas,scripts/'score_answer_quality_ragas.py','--batch-dir','evaluation/artifacts/answer_quality_v1',
                    '--metric','faithfulness','--language','zh','--schemes-filter','vector','--question-ids',qid,
                    '--judge-max-tokens','8192','--workers','1','--allow-paid'])
        run('offline_comparison',[backend,scripts/'build_faithfulness_comparison.py'])
        print(json.dumps(read(OUTPUT/'comparison_report.json'),ensure_ascii=False,indent=2))
    finally:task_lock.unlink(missing_ok=True)


if __name__=='__main__':main()
