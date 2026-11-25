"""固定24条技术FAQ：原始向量回答与改进回答各一条，不混用旧问法分数。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import random
import sys

from generate_grounded_answers import ROOT,PROMPT,digest,write,parse_grounded


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-run',required=True);parser.add_argument('--improved-run',required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--contextual',action='store_true');args=parser.parse_args()
    active_prompt=PROMPT
    if args.contextual:
        active_prompt+='''\n回答表达补充：每条结论应交代正在解决的对象或触发条件，再给对应结论，让人脱离问题单看答案也能知道它在解决什么。
保留问题中的关键报错或场景术语，解释它与参数、操作的对应关系；不要只给参数值而遗漏该排查什么。
多部分问题逐项回答，先回答问的内容再给必要限定。禁止照抄整道问题、用问题作标题或增加无关建议。'''
    data_path=ROOT/'evaluation/datasets/enterprise_quality_faq_v1/questions.json'
    if digest(data_path)!='eaa065ef12df0807916e24bb913013e774120db48bb22d20485b3d0dd2efb487':raise ValueError('FAQ changed')
    questions=json.loads(data_path.read_text(encoding='utf-8'))['questions']
    before_path=(ROOT/args.baseline_run/'results.jsonl').resolve();after_path=(ROOT/args.improved_run/'results.jsonl').resolve()
    for p in (before_path,after_path):
        if not p.is_relative_to((ROOT/'evaluation/artifacts').resolve()):raise ValueError('Source outside artifacts')
    before={r['question_id']:r for r in map(json.loads,before_path.read_text(encoding='utf-8').splitlines())}
    after={r['question_id']:r for r in map(json.loads,after_path.read_text(encoding='utf-8').splitlines())}
    if set(before)!=set(after) or len(before)!=24:raise ValueError('Retrieval batch mismatch')
    old_manifest=json.loads((ROOT/'evaluation/artifacts/answer_quality_v1/manifest.json').read_text(encoding='utf-8'))
    old_prompt=old_manifest['project_system_prompt']+'\n\n'+old_manifest['final_stage_instruction']
    sys.path.insert(0,str(ROOT/'backend'))
    from app.services.chat_service import ChatService
    jobs=[]
    for q in questions:
        qid=q['id']
        if before[qid]['question']!=q['question'] or after[qid]['question']!=q['question']:raise ValueError('Query differs')
        if before[qid]['results']['vector']['status']!='ok' or after[qid]['status']!='ok':raise ValueError('Retrieval has errors')
        for scheme,hits in [('vector',before[qid]['results']['vector']['hits']),('grounded_final',after[qid]['hits'])]:
            hits=hits[:5]
            if scheme=='vector':
                bodies=[h['content'][:500] for h in hits]
                blocks=[ChatService._format_prefetched_context(None,[h]).replace('[1] filename=',f'[{i}] filename=',1) for i,h in enumerate(hits,1)]
            else:
                bodies=[h['content'] for h in hits]
                blocks=[f'[{i}] filename={h["filename"]}\n'+(('section_title='+str(h.get('section_title') or '')+'\n') if args.contextual else '')+body for i,(h,body) in enumerate(zip(hits,bodies),1)]
            jobs.append({'sample_id':qid+':'+scheme,'question_id':qid,'scheme':scheme,'user_input':q['question'],
                         'reference':q['reference_answer'],'source_chunk_ids':[h['chunk_id'] for h in hits],
                         'context_contents':bodies,'retrieved_contexts':blocks,'rendered_context':'\n\n'.join(blocks)})
    random.Random(20261123).shuffle(jobs)
    dest=ROOT/('evaluation/artifacts/answer_quality_faq_contextual_v2' if args.contextual else 'evaluation/artifacts/answer_quality_faq_v1');dest.mkdir(exist_ok=True)
    baseline_cache={}
    if args.contextual:
        cache=ROOT/'evaluation/artifacts/answer_quality_faq_v1_validated/answers.jsonl'
        baseline_cache={r['sample_id']:r for r in map(json.loads,cache.read_text(encoding='utf-8').splitlines()) if r['scheme']=='vector'}
        if len(baseline_cache)!=24:raise ValueError('Cached baseline incomplete')
    if (dest/'ready.json').exists():
        ready=json.loads((dest/'ready.json').read_text(encoding='utf-8'));assert digest(dest/'answers.jsonl')==ready['answers_sha256']
        print('FAQ answers already complete; no paid calls repeated.');return
    if (dest/'answers.jsonl').exists():raise ValueError('Partial generation exists; preserve logs and stop')
    write(dest/'input_jobs.json',jobs)
    manifest={'status':'prepared','question_count':24,'answer_count':48,'schemes':['vector','grounded_final'],
              'dataset_sha256':digest(data_path),'baseline_results_sha256':digest(before_path),'improved_results_sha256':digest(after_path),
              'baseline_run':args.baseline_run,'improved_run':args.improved_run,
              'baseline_model':'qwen-plus','improved_model':'qwen3.7-max','baseline_prompt':old_prompt,'improved_prompt':active_prompt,
              'reference_or_gold_sent_to_generator':False,'input_jobs_sha256':digest(dest/'input_jobs.json'),
              'context_rule':'baseline actual project prefix500; improved full unchanged Top5 real bodies',
              'scope':'24 concise enterprise technical FAQ development tasks; same frozen questions',
              'baseline_reuse':bool(baseline_cache),'contextual_answer_policy':args.contextual}
    write(dest/'manifest.json',manifest)
    if args.prepare_only:print('Prepared48 jobs; no model calls.');return
    if not args.allow_paid:parser.error('Requires --allow-paid')
    from app.config import get_settings
    import dashscope
    settings=get_settings()
    lock=dest/'generation.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        manifest['status']='running';write(dest/'manifest.json',manifest)
        def generate(job):
            row=dict(job);improved=job['scheme']=='grounded_final';model='qwen3.7-max' if improved else 'qwen-plus'
            if job['sample_id'] in baseline_cache:
                cached=baseline_cache[job['sample_id']]
                for key,value in job.items():
                    if cached[key]!=value:raise ValueError('Baseline cache input mismatch')
                return {**cached,'reused_baseline':True}
            try:
                r=dashscope.Generation.call(model=model,api_key=settings.dashscope_api_key,result_format='message',temperature=0,
                    enable_thinking=False,max_tokens=1800 if improved else 1024,
                    messages=[{'role':'system','content':active_prompt if improved else old_prompt},
                              {'role':'system','content':'本次已检索到的候选资料：\n'+job['rendered_context']},
                              {'role':'user','content':job['user_input']}])
                usage=getattr(r,'usage',None) or {}
                row.update(model=model,status_code=r.status_code,request_id=r.request_id,
                    usage={k:usage[k] for k in ('total_tokens','input_tokens','output_tokens') if usage.get(k) is not None})
                if r.status_code!=200:raise RuntimeError('Generation API error')
                raw=r.output['choices'][0]['message']['content'];row['raw_response']=raw
                if improved:
                    if args.contextual:
                        from validate_faq_batch import expand_excerpt
                        obj=json.loads(raw.strip())
                        for statement in obj['statements']:
                            quote,changed=expand_excerpt(statement['quote'],job['context_contents'][statement['source']-1])
                            if changed:statement['original_ellipsis_quote']=statement['quote'];statement['quote']=quote
                        parsed=json.dumps(obj,ensure_ascii=False)
                    else:parsed=raw
                    obj,answer=parse_grounded(parsed,job,allow_visible_markdown=True)
                    row.update(response=answer,evidence_statements=obj['statements'],insufficient_evidence=obj['insufficient_evidence'],missing=obj.get('missing',''))
                else:row['response']=raw.strip()
                if not row['response']:raise ValueError('Empty answer')
                row.update(status='ok',finish_reason=r.output['choices'][0].get('finish_reason'))
            except Exception as exc:
                row.update(status='error',response=None,error_type=type(exc).__name__,error_detail=str(exc).replace(settings.dashscope_api_key,'[REDACTED]')[:300])
            return row
        records=[]
        with (dest/'answers.jsonl').open('w',encoding='utf-8') as stream,ThreadPoolExecutor(max_workers=3) as executor:
            pending=[executor.submit(generate,job) for job in jobs]
            for future in as_completed(pending):
                row=future.result();records.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'sample':row['sample_id'],'status':row['status'],'error':row.get('error_detail')},ensure_ascii=False),flush=True)
        manifest.update(status='complete' if all(r['status']=='ok' for r in records) else 'stopped_with_errors',
                        errors=sum(r['status']!='ok' for r in records),refusals=sum(r.get('insufficient_evidence',False) for r in records),
                        reused_baseline_answers=sum(r.get('reused_baseline',False) for r in records),
                        new_model_calls=sum(not r.get('reused_baseline',False) for r in records))
        write(dest/'manifest.json',manifest)
        if manifest['errors']:raise ValueError('Generation errors: preserve all rows, do not filter or automatically retry')
        write(dest/'ready.json',{'status':'ready','answers':48,'questions':24,'schemes':['vector','grounded_final'],
            'answers_sha256':digest(dest/'answers.jsonl'),'manifest_sha256':digest(dest/'manifest.json')})
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
