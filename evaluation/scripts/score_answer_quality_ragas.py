"""固定Ragas0.4.4指标评分；相关性和忠实度可在不同进程并行。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import contextvars
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import threading
from unittest.mock import patch

os.environ['RAGAS_DO_NOT_TRACK']='true'
os.environ['LANGCHAIN_TRACING_V2']='false'
os.environ['LANGSMITH_TRACING']='false'
ROOT=Path(__file__).resolve().parents[2]
SAMPLE_ID=contextvars.ContextVar('quality_sample_id',default=None)


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metric',choices=('relevancy','faithfulness'),required=True)
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--batch-dir',default='evaluation/artifacts/answer_quality_v1')
    parser.add_argument('--language',choices=['default','zh'],default='default')
    parser.add_argument('--schemes-filter',default='');parser.add_argument('--question-limit',type=int,default=0)
    parser.add_argument('--question-ids',default='');parser.add_argument('--judge-max-tokens',type=int,default=3072)
    parser.add_argument('--embedding-model',choices=['text-embedding-v1','text-embedding-v4'],default='text-embedding-v1');args=parser.parse_args()
    if not 1<=args.workers<=6:parser.error('workers must be between 1 and 6')
    if not args.allow_paid:parser.error('Metric judging uses paid model APIs; requires --allow-paid')
    if importlib.metadata.version('ragas')!='0.4.4':raise ValueError('Requires pinned ragas0.4.4 in isolated evaluation env')
    from dotenv import dotenv_values
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_community.embeddings import DashScopeEmbeddings
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import ResponseRelevancy,Faithfulness
    from ragas.run_config import RunConfig
    directory=(ROOT/args.batch_dir).resolve()
    if not directory.is_relative_to((ROOT/'evaluation/artifacts').resolve()):raise ValueError('Batch must be within evaluation/artifacts')
    ready=json.loads((directory/'ready.json').read_text(encoding='utf-8'))
    if hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()!=ready['manifest_sha256']:
        raise ValueError('Frozen generation manifest changed')
    answers_path=directory/'answers.jsonl'
    if hashlib.sha256(answers_path.read_bytes()).hexdigest()!=ready['answers_sha256']:raise ValueError('Frozen answers changed')
    answers=sorted([json.loads(line) for line in answers_path.read_text(encoding='utf-8').splitlines()],key=lambda row:row['sample_id'])
    if len(answers)!=ready['answers'] or len({row['sample_id'] for row in answers})!=ready['answers'] or any(row['status']!='ok' for row in answers):
        raise ValueError('Answer batch incomplete')
    schemes=ready.get('schemes',['bm25','vector','final_chain'])
    if set(row['scheme'] for row in answers)!=set(schemes):raise ValueError('Scheme mismatch')
    if args.schemes_filter:
        selected=args.schemes_filter.split(',')
        if not set(selected).issubset(schemes):raise ValueError('Unknown scheme filter')
        schemes=selected;answers=[row for row in answers if row['scheme'] in schemes]
    if args.question_limit:
        qids=set(sorted({row['question_id'] for row in answers})[:args.question_limit])
        answers=[row for row in answers if row['question_id'] in qids]
    if args.question_ids:
        qids=set(args.question_ids.split(','))
        if not qids.issubset({row['question_id'] for row in answers}):raise ValueError('Unknown requested question ID')
        answers=[row for row in answers if row['question_id'] in qids]
    if args.limit:answers=answers[:args.limit]
    suffix=('_zh2' if args.language=='zh' else '')
    suffix+=(f'_qfirst{args.question_limit}' if args.question_limit else '')
    suffix+=('_'+args.schemes_filter.replace(',','-') if args.schemes_filter else '')
    suffix+=('_ids'+args.question_ids.replace(',','-') if args.question_ids else '')
    suffix+=(f'_max{args.judge_max_tokens}' if args.judge_max_tokens!=3072 else '')
    suffix+=('_embv4' if args.embedding_model=='text-embedding-v4' else '')
    destination=directory/(args.metric+'_qwen37max_v1'+suffix+(f'_pilot{args.limit}' if args.limit else ''))
    destination.mkdir(exist_ok=args.resume)
    previous_manifest=None
    if args.resume:
        previous_manifest=json.loads((destination/'manifest.json').read_text(encoding='utf-8'))
        if previous_manifest['status'].startswith('complete'):
            print('Scoring already complete; no paid calls repeated');return
        if previous_manifest['answers_sha256']!=ready['answers_sha256'] or previous_manifest['judge_model']!='qwen3.7-max':
            raise ValueError('Resume settings mismatch')
    values=dotenv_values(ROOT/'backend/.env')
    key=os.environ.get('DASHSCOPE_API_KEY') or values.get('DASHSCOPE_API_KEY')
    if not key:raise ValueError('Local model key is missing')
    judge=ChatTongyi(model_name='qwen3.7-max',dashscope_api_key=key,streaming=False,max_retries=1,
                    model_kwargs={'enable_thinking':False,'max_tokens':args.judge_max_tokens})
    score_timeout=300 if args.judge_max_tokens>3072 else 180
    config=RunConfig(timeout=score_timeout,max_retries=1,max_workers=2)
    llm=LangchainLLMWrapper(judge,run_config=config)
    embeddings=None
    if args.metric=='relevancy':
        embeddings=DashScopeEmbeddings(model=args.embedding_model,dashscope_api_key=key)
        metric=ResponseRelevancy(llm=llm,embeddings=LangchainEmbeddingsWrapper(embeddings,run_config=config),strictness=3)
    else:metric=Faithfulness(llm=llm)
    if args.language=='zh':
        from ragas.metrics._answer_relevance import ResponseRelevanceInput,ResponseRelevanceOutput
        from ragas.metrics._faithfulness import StatementGeneratorInput,StatementGeneratorOutput
        if args.metric=='relevancy':
            prompt=copy.deepcopy(metric.question_generation)
            prompt.instruction='根据给定回答，反推出一个该回答能够回答的问题，使用简体中文并保留必要技术名称。判断回答是否回避、含糊或不确定；若是则noncommittal=1，否则为0。例如“我不知道”或“我不确定”属于不确定回答。仅输出规定的JSON。'
            prompt.examples=[(ResponseRelevanceInput(response='爱因斯坦出生于德国。'),ResponseRelevanceOutput(question='爱因斯坦出生在哪里？',noncommittal=0)),
                             (ResponseRelevanceInput(response='我不知道那部新手机的主要创新，因为没有相关资料。'),ResponseRelevanceOutput(question='那部新手机的主要创新是什么？',noncommittal=1))]
            prompt.language='chinese';metric.question_generation=prompt
        else:
            prompt=copy.deepcopy(metric.statement_generator_prompt)
            prompt.instruction='给定一个问题和回答，将回答中的每个句子拆解成一个或多个可独立理解的事实陈述，使用简体中文，不使用代词。只拆解回答实际表达的陈述，不将问题中的额外场景或模型背景知识添入陈述。输出规定的JSON。'
            prompt.examples=[(StatementGeneratorInput(question='爱因斯坦是谁？',answer='他是出生于德国的理论物理学家，提出了相对论。'),StatementGeneratorOutput(statements=['爱因斯坦是出生于德国的理论物理学家。','爱因斯坦提出了相对论。']))]
            prompt.language='chinese';metric.statement_generator_prompt=prompt
            prompt=copy.deepcopy(metric.nli_statements_prompt)
            prompt.instruction='仅根据给定上下文判断每条陈述是否忠实。能够从上下文直接推断的陈述，verdict=1；不能直接推断或被上下文否定的陈述，verdict=0。使用简体中文说明理由，逐字保留输入statement，输出规定的JSON。'
            prompt.language='chinese';metric.nli_statements_prompt=prompt
    metric.init(config)
    manifest={'status':'running','metric':args.metric,'ragas_version':'0.4.4','api':'legacy SingleTurnSample / single_turn_ascore',
              'judge_model':'qwen3.7-max','embedding_model':args.embedding_model if embeddings else None,
              'strictness':3 if embeddings else None,'answers_sha256':ready['answers_sha256'],'samples':len(answers),
              'reference_answers_sent_to_judge':False,'scheme_label_separately_sent':False,
              'context_rule':'exact rendered context actually supplied to generator; see frozen generation manifest for body limits',
              'score_clipping':False,'postprocess':'report undefined/errors and finite-score coverage explicitly',
              'independent_external_human_review':False,'workers':args.workers}
    manifest.update(prompt_language=args.language,question_limit=args.question_limit,scheme_filter=args.schemes_filter)
    manifest.update(question_ids=args.question_ids,judge_max_tokens=args.judge_max_tokens)
    if args.language=='zh':
        manifest['localized_prompt_instructions']={name:getattr(metric,name).instruction for name in
            (['question_generation'] if args.metric=='relevancy' else ['statement_generator_prompt','nli_statements_prompt'])}
    if previous_manifest:
        checkpoint=destination/'manifest_before_resume.json'
        if not checkpoint.exists():write_json(checkpoint,previous_manifest)
        manifest['resumed_from_checkpoint']=True
        manifest['interrupted_inflight_calls_may_have_unreported_usage']=True
    write_json(destination/'manifest.json',manifest)
    lock=threading.Lock();events=[];abort=threading.Event()
    if args.resume and (destination/'api_events.json').exists():
        events=json.loads((destination/'api_events.json').read_text(encoding='utf-8'))
        for event in events:
            if event['status']=='started':event['status']='unknown_after_interruption'

    def tracked(original,kind):
        def call(**kwargs):
            if abort.is_set():raise RuntimeError('Model access failure; stopping further API calls')
            event={'sample_id':SAMPLE_ID.get(),'kind':kind,'model':kwargs.get('model'),'status':'started'}
            with lock:events.append(event)
            try:
                response=original(**kwargs)
                usage=getattr(response,'usage',None) or {}
                event.update(status='returned',status_code=getattr(response,'status_code',None),
                             request_id=getattr(response,'request_id',None),usage={k:usage[k] for k in ('total_tokens','input_tokens','output_tokens') if usage.get(k) is not None})
                if event['status_code'] in (400,401,403,404):abort.set()
                if kind=='judge' and event['status_code']==200:
                    event['response_text']=response.output['choices'][0]['message']['content']
                return response
            except Exception as exc:
                event.update(status='exception',error_type=type(exc).__name__);raise
            finally:
                with lock:write_json(destination/'api_events.json',events)
        return call

    semaphore=asyncio.Semaphore(args.workers)
    async def score(row):
        async with semaphore:
            token=SAMPLE_ID.set(row['sample_id'])
            record={'sample_id':row['sample_id'],'question_id':row['question_id'],'scheme':row['scheme'],'metric':args.metric}
            try:
                sample=SingleTurnSample(user_input=row['user_input'],response=row['response'],retrieved_contexts=row['retrieved_contexts'])
                value=float(await metric.single_turn_ascore(sample,timeout=score_timeout))
                record.update(score=value if math.isfinite(value) else None,status='ok' if math.isfinite(value) else 'undefined')
            except Exception as exc:
                record.update(score=None,status='error',error_type=type(exc).__name__,
                              error_detail=str(exc).replace(key,'[REDACTED]')[:600])
            finally:SAMPLE_ID.reset(token)
            return record

    results=[]
    scores_path=destination/'scores.jsonl'
    if args.resume and scores_path.exists():
        results=[json.loads(line) for line in scores_path.read_text(encoding='utf-8').splitlines()]
    completed={row['sample_id'] for row in results}
    if len(completed)!=len(results) or not completed.issubset({row['sample_id'] for row in answers}):raise ValueError('Invalid resume checkpoint')
    manifest['previously_scored_samples']=len(results)
    write_json(destination/'manifest.json',manifest)
    with ExitStack() as stack,scores_path.open('a' if args.resume else 'w',encoding='utf-8') as stream:
        stack.enter_context(patch.object(judge.client,'call',side_effect=tracked(judge.client.call,'judge')))
        if embeddings:stack.enter_context(patch.object(embeddings.client,'call',side_effect=tracked(embeddings.client.call,'embedding')))
        pending=[asyncio.create_task(score(row)) for row in answers if row['sample_id'] not in completed]
        for task in asyncio.as_completed(pending):
            row=await task;results.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={}
    for scheme in schemes:
        rows=[row for row in results if row['scheme']==scheme];finite=[row['score'] for row in rows if row['score'] is not None]
        summary[scheme]={'samples':len(rows),'finite_scores':len(finite),'undefined':sum(row['status']=='undefined' for row in rows),
                         'errors':sum(row['status']=='error' for row in rows),'mean_on_finite':sum(finite)/len(finite) if finite else None,
                         'finite_score_fraction':len(finite)/len(rows) if rows else None,
                         'mean_with_missing_as_zero_sensitivity':sum(finite)/len(rows) if rows else None}
    manifest.update(status='complete' if all(row['status']=='ok' for row in results) else 'complete_with_undefined_or_errors',
                    api_calls=len(events),reported_tokens=sum(event.get('usage',{}).get('total_tokens',0) for event in events),
                    judge_input_tokens=sum(event.get('usage',{}).get('input_tokens',0) for event in events if event['kind']=='judge'),
                    judge_output_tokens=sum(event.get('usage',{}).get('output_tokens',0) for event in events if event['kind']=='judge'),
                    embedding_tokens=sum(event.get('usage',{}).get('total_tokens',0) for event in events if event['kind']=='embedding'))
    write_json(destination/'summary.json',summary);write_json(destination/'manifest.json',manifest)
    print(json.dumps({'output':str(destination),'summary':summary},ensure_ascii=False,indent=2))


if __name__=='__main__':
    logging.disable(logging.CRITICAL);asyncio.run(main())
