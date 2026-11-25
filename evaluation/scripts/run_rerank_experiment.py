"""固定语义题：扩大候选的RRF与内容重排对照。答案仅用于返回后评分。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
from datetime import datetime,timedelta,timezone
from time import perf_counter
from unittest.mock import patch

from enterprise_runtime import ROOT,digest,write_json,isolated_settings,validate_sources
from enterprise_evidence_scoring import score_hits
from run_retrieval import check_postgres,check_vectors,validate_hits,sanitize,track_embedding_call
from app.rag.retriever import _vector_retrieve_chunks,_bm25_retrieve_chunks,_fuse_hits
from app.core.embeddings import get_embeddings
import dashscope


def parse_rerank(response,candidates):
    if response.status_code!=200:
        raise RuntimeError('Rerank API non-success status')
    rows=response.output['results']
    indices=[row['index'] for row in rows]
    if len(indices)!=len(set(indices)) or any(not isinstance(i,int) or i<0 or i>=len(candidates) for i in indices):
        raise ValueError('Invalid rerank indices')
    hits=[]
    for row in rows:
        score=float(row['relevance_score'])
        if not math.isfinite(score):raise ValueError('Nonfinite rerank score')
        hits.append({**candidates[row['index']],'rerank_score':score,'rerank_input_index':row['index']})
    return hits


def summarize(records,modes):
    output={}
    for mode in modes:
        results=[r['results'][mode] for r in records]
        full=sum(result['evidence_score']['complete_evidence_at_k'] for result in results)
        output[mode]={'questions':len(results),'complete_hits':full,'complete_evidence_at_5':full/len(results),
                      'execution_errors':sum(result['status']!='ok' for result in results)}
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--candidate-k',type=int,default=50)
    args=parser.parse_args()
    if not args.allow_paid:parser.error('Paid embedding/rerank requires --allow-paid')
    if args.candidate_k not in (50,100):raise ValueError('Candidate count must be explicitly 50 or100')
    validate_sources();_,settings=isolated_settings()
    dataset_path=ROOT/'evaluation/datasets/enterprise_semantic_paraphrase_v1/questions.json'
    if digest(dataset_path)!='4f4549d7002d07371f31e474e5869f807183cdde52a41fbea4ebb8f3c54aa10f':
        raise ValueError('Fixed semantic queries changed')
    dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    snapshot_path=ROOT/dataset['ingestion_snapshot']
    if digest(snapshot_path)!=dataset['ingestion_snapshot_sha256']:raise ValueError('Snapshot changed')
    snapshot=json.loads(snapshot_path.read_text(encoding='utf-8'))
    check_postgres(snapshot);check_vectors(snapshot,settings)
    expected={chunk['id']:chunk for chunk in snapshot['chunks']}
    now=datetime.now(timezone(timedelta(hours=8)));out=ROOT/'evaluation/artifacts'/f'enterprise_rerank_k{args.candidate_k}_{now:%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    manifest={'started_at':now.isoformat(),'status':'running','scope':'development on24 exposed fixed semantic questions',
              'dataset_sha256':digest(dataset_path),'candidate_k':args.candidate_k,'top_k':5,'rrf_k':60,
              'rerank_model':'qwen3-rerank','rerank_input':'full deduplicated candidate union in RRF order; title+body',
              'embedding_model':settings.embedding_model,'no_gold_or_reference_answer_sent':True,
              'code_sha256':digest(ROOT/'evaluation/scripts/run_rerank_experiment.py'),'pre_run_data_check':'passed',
              'post_run_data_check':'pending','price_rmb_per_million_tokens':{'embedding':0.7,'rerank':0.5}}
    write_json(out/'manifest.json',manifest)
    events,embed_events,records=[],[],[]
    modes=('vector','bm25','rrf','rrf_rerank')
    with (out/'results.jsonl').open('w',encoding='utf-8') as stream,patch.object(get_embeddings().client,'call',
        side_effect=track_embedding_call(get_embeddings().client.call,embed_events)):
        for question in dataset['questions']:
            started=perf_counter();candidates={};errors={}
            for source,function in [('vector',_vector_retrieve_chunks),('bm25',_bm25_retrieve_chunks)]:
                try:
                    hits=function(query=question['question'],candidate_k=args.candidate_k,raise_on_error=True)
                    if len(hits)>args.candidate_k:raise ValueError('Too many candidates')
                    validate_hits(hits,expected);candidates[source]=hits
                except Exception as exc:
                    candidates[source]=[];errors[source]=type(exc).__name__
            fused=_fuse_hits(query=question['question'],vector_hits=copy.deepcopy(candidates['vector']),
                             bm25_hits=copy.deepcopy(candidates['bm25']),top_k=args.candidate_k*2)
            ranked=[];rerank_error=None
            event={'question_id':question['id'],'model':'qwen3-rerank','documents':len(fused),'status':'started'}
            try:
                if not fused:raise ValueError('No candidates to rerank')
                texts=[str(hit.get('section_title') or '')+'\n'+hit['content'] for hit in fused]
                response=dashscope.TextReRank.call(model='qwen3-rerank',api_key=settings.dashscope_api_key,
                    query=question['question'],documents=texts,top_n=5,return_documents=False)
                usage=getattr(response,'usage',None) or {}
                event.update(status='returned',status_code=response.status_code,request_id=response.request_id,
                             response_code=getattr(response,'code',None),
                             usage={key:usage[key] for key in ('total_tokens','input_tokens','output_tokens') if usage.get(key) is not None})
                ranked=parse_rerank(response,fused)
                if len(ranked)!=min(5,len(fused)):raise ValueError('Rerank result count differs')
            except Exception as exc:
                rerank_error=type(exc).__name__;event['error_type']=rerank_error
            events.append(event)
            results={}
            for mode,hits in [('vector',candidates['vector'][:5]),('bm25',candidates['bm25'][:5]),('rrf',fused[:5]),('rrf_rerank',ranked[:5])]:
                error=errors.get(mode) if mode in ('vector','bm25') else (rerank_error if mode=='rrf_rerank' else None)
                if mode in ('rrf','rrf_rerank') and errors:error=error or 'CandidateRouteError'
                results[mode]={'status':'ok' if not error else 'error','error_type':error,
                               'hits':sanitize(hits),'evidence_score':score_hits(question['evidence_mapping'],[h['chunk_id'] for h in hits])}
            record={'question_id':question['id'],'question':question['question'],'reference_answer':question['reference_answer'],
                    'evidence_mapping':question['evidence_mapping'],'candidate_errors':errors,'candidates':sanitize(candidates),
                    'full_fused_candidates':sanitize(fused),'results':results,'rerank_event':event,
                    'total_component_ms':(perf_counter()-started)*1000}
            records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            write_json(out/'rerank_events.json',events);write_json(out/'embedding_events.json',embed_events)
            print(json.dumps({'id':question['id'],'hits':{mode:result['evidence_score']['complete_evidence_at_k'] for mode,result in results.items()},
                              'rerank_status':event.get('status_code')},ensure_ascii=False),flush=True)
            if event.get('status_code') in (401,403,404):
                raise RuntimeError('Rerank access/model unavailable; stopped further paid attempts')
    check_postgres(snapshot);check_vectors(snapshot,settings)
    summary=summarize(records,modes)
    manifest.update(status='complete',post_run_data_check='passed',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    embedding_sdk_calls=len(embed_events),rerank_sdk_calls=len(events),
                    embedding_tokens=sum(e.get('usage',{}).get('total_tokens',0) for e in embed_events),
                    rerank_tokens=sum(e.get('usage',{}).get('total_tokens',e.get('usage',{}).get('input_tokens',0)) for e in events))
    manifest['estimated_cost_rmb']=(manifest['embedding_tokens']*0.7+manifest['rerank_tokens']*0.5)/1e6
    write_json(out/'summary.json',summary);write_json(out/'manifest.json',manifest)
    print(json.dumps({'output':out.relative_to(ROOT).as_posix(),'summary':summary,'estimated_cost_rmb':manifest['estimated_cost_rmb'],
                      'embedding_tokens':manifest['embedding_tokens'],'rerank_tokens':manifest['rerank_tokens']},ensure_ascii=False,indent=2))
    return int(any(result['status']!='ok' for record in records for result in record['results'].values()))


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'Rerank experiment stopped: {type(exc).__name__}',file=sys.stderr);raise SystemExit(2)
