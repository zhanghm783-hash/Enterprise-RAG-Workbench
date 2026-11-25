"""固定企业题的三路真实对照；使用项目候选与RRF，不使用PG兜底。"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from enterprise_runtime import ROOT, CONTRACT, CONTRACT_SHA, isolated_settings, validate_sources, digest, write_json
from enterprise_evidence_scoring import score_hits
from evaluation_datasets import load_dataset
from run_retrieval import (
    run_question, check_postgres, check_vectors, track_embedding_call,
    _vector_retrieve_chunks, _bm25_retrieve_chunks, _candidate_k, _RRF_K, get_embeddings,
)


def score_record(record, mapping):
    for result in record['results'].values():
        result['evidence_score'] = score_hits(mapping, [hit['chunk_id'] for hit in result['hits']], k=5)
        result['legacy_any_gold_hit_at_5'] = result.pop('hit_at_5')
    record['evidence_mapping'] = mapping
    return record


def summarize(records):
    summary = {}
    for mode in ('vector', 'bm25', 'hybrid'):
        entries = [record['results'][mode] for record in records]
        full = sum(entry['evidence_score']['complete_evidence_at_k'] for entry in entries)
        partial = sum(entry['evidence_score']['any_evidence_hit_at_k'] for entry in entries)
        summary[mode] = {'questions':len(entries), 'complete_evidence_hits':full,
                         'complete_evidence_at_5':full/len(entries) if entries else None,
                         'any_evidence_hits':partial, 'any_evidence_hit_at_5':partial/len(entries) if entries else None,
                         'statuses':dict(Counter(entry['status'] for entry in entries))}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev','test'), default='dev')
    parser.add_argument('--dataset', choices=('enterprise','semantic','faq'), default='enterprise')
    parser.add_argument('--allow-embedding', action='store_true')
    args = parser.parse_args(argv)
    if not args.allow_embedding:
        parser.error('Query embedding requires --allow-embedding')
    validate_sources()
    dataset_path,dataset_hash,dataset=load_dataset(args.dataset)
    if dataset['status'] != 'frozen' or dataset['scoring_contract_sha256'] != CONTRACT_SHA:
        raise ValueError('Frozen contract mismatch')
    if args.dataset=='semantic':
        parent_path=ROOT/dataset['parent_frozen_dataset']
        if digest(parent_path)!=dataset['parent_frozen_dataset_sha256']:
            raise ValueError('Semantic parent dataset changed')
        parent=json.loads(parent_path.read_text(encoding='utf-8'))
        original={q['id']:q for q in parent['questions'] if q['split']=='test'}
        if {q['id'] for q in dataset['questions']}!=set(original):
            raise ValueError('Semantic question pairs were filtered')
        for q in dataset['questions']:
            old=original[q['id']]
            if q['reference_answer']!=old['reference_answer'] or q['evidence_mapping']!=old['evidence_mapping'] or q['gold_chunk_ids']!=old['gold_chunk_ids']:
                raise ValueError('Semantic answers or gold IDs changed')
    snapshot_path = ROOT/dataset['ingestion_snapshot']
    if digest(snapshot_path) != dataset['ingestion_snapshot_sha256']:
        raise ValueError('Ingestion snapshot changed')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
    rules = {q['id']:q for q in contract['questions']}
    questions = [q for q in dataset['questions'] if q['split'] == args.split]
    if len(questions) != dataset['counts'][args.split]:
        raise ValueError('Frozen question count mismatch')
    for q in questions:
        if q['evidence_mapping']['missing_required_units']:
            raise ValueError('Required gold label missing')
    _, settings = isolated_settings()
    if settings.milvus_collection != snapshot['local_settings']['milvus_collection']:
        raise ValueError('Collection differs from snapshot')
    check_postgres(snapshot)
    check_vectors(snapshot, settings)
    expected = {chunk['id']:chunk for chunk in snapshot['chunks']}
    now = datetime.now(timezone(timedelta(hours=8)))
    prefix={'enterprise':'enterprise','semantic':'enterprise_semantic','faq':'enterprise_faq'}[args.dataset]
    output = ROOT/'evaluation/artifacts'/f'{prefix}_{args.split}_{now:%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'running','started_at':now.isoformat(),'split':args.split,
                'dataset_path':dataset_path.relative_to(ROOT).as_posix(),'dataset_id':dataset['dataset_id'],
                'dataset_sha256':dataset_hash,'snapshot_sha256':digest(snapshot_path),'contract_sha256':CONTRACT_SHA,
                'top_k':5,'candidate_k':_candidate_k(5),'rrf_k':_RRF_K,'question_count':len(questions),
                'database':'rag_enterprise_v2','collection':settings.milvus_collection,
                'postgres_fallback':False,'embedding_model':settings.embedding_model,
                'shared_candidates_across_modes':True,'code_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in
                sorted((ROOT/'backend/app').rglob('*.py'))+sorted((ROOT/'evaluation/scripts').glob('*.py'))},
                'pre_run_data_check':'passed','post_run_data_check':'pending',
                'timing_note':'Sequential reused component timings, not live concurrent web latency'}
    write_json(output/'manifest.json',manifest)
    records, events = [], []
    with ExitStack() as stack:
        client = get_embeddings().client
        stack.enter_context(patch.object(client,'call',side_effect=track_embedding_call(client.call,events)))
        stream = stack.enter_context((output/'results.jsonl').open('w',encoding='utf-8'))
        for question in questions:
            # 候选函数只接收问题和候选数；答案与标签只用于返回后的评分。
            adapted = copy.deepcopy(question)
            adapted['baseline_full_evidence_chunks'] = [{'chunk_id':qid} for qid in question['gold_chunk_ids']]
            adapted['evidence_options'] = rules[question['id']]['required_units']
            first_event = len(events)
            record = run_question(adapted,modes=['vector','bm25','hybrid'],expected=expected,
                                  vector_fn=_vector_retrieve_chunks,bm25_fn=_bm25_retrieve_chunks)
            record = score_record(record,question['evidence_mapping'])
            record['embedding_sdk_events'] = copy.deepcopy(events[first_event:])
            records.append(record)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            write_json(output/'embedding_sdk_events.json',events)
            print(json.dumps({'id':question['id'],'complete_evidence_at_5':{mode:result['evidence_score']['complete_evidence_at_k']
                  for mode,result in record['results'].items()},'statuses':{mode:result['status'] for mode,result in record['results'].items()}},ensure_ascii=False),flush=True)
    try:
        check_postgres(snapshot)
        check_vectors(snapshot,settings)
        manifest['post_run_data_check']='passed'
    except Exception as exc:
        manifest['post_run_data_check']=f'failed:{type(exc).__name__}'
    summary=summarize(records)
    manifest.update(status='complete',finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    embedding_sdk_calls=len(events),reported_total_tokens=sum(event.get('usage',{}).get('total_tokens',0) for event in events))
    manifest['estimated_cost_rmb']=manifest['reported_total_tokens']*0.7/1e6
    manifest['usage_note']='SDK reported usage; internal HTTP attempts and account bill not independently inspected'
    write_json(output/'summary.json',summary);write_json(output/'manifest.json',manifest)
    print(json.dumps({'output':output.relative_to(ROOT).as_posix(),'summary':summary,'tokens':manifest['reported_total_tokens'],
                      'estimated_cost_rmb':manifest['estimated_cost_rmb'],'post_run_data_check':manifest['post_run_data_check']},ensure_ascii=False,indent=2))
    return int(manifest['post_run_data_check']!='passed' or any(result['status']!='ok' for record in records for result in record['results'].values()))


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Enterprise evaluation stopped: {type(exc).__name__}',file=sys.stderr)
        raise SystemExit(2)
