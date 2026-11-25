"""同20文件真实入库对照；仅使用专用本机数据库及Milvus集合。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from complex_document_quality import RULES, evaluate

ROOT=Path(__file__).resolve().parents[2]
CORPUS=ROOT/'evaluation/corpus/complex_enterprise_v1'
ART=ROOT/'evaluation/artifacts'
sys.path.insert(0,str(ROOT/'backend'))
os.environ['OCR_ENABLED']='false'

from app.config import get_settings
from app.core.postgres import _normalize_sqlalchemy_dsn,get_engine,get_session_factory
from app.core.embeddings import get_embeddings
from app.models import Base,Chunk,Document
from app.rag.loader import LoadedDocument,load_document,_split_markdown_sections
from app.rag.ingest import ingest_loaded_document
from enterprise_runtime import snapshot_pg
from run_retrieval import check_vectors,track_embedding_call
from sqlalchemy import create_engine,select,text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session
from pymilvus import MilvusClient


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['native','enhanced'],required=True)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--allow-paid',action='store_true')
    ap.add_argument('--ocr-dir',default='complex_document_ocr_v1');args=ap.parse_args()
    if not args.prepare_only and not args.allow_paid:ap.error('Real embeddings require --allow-paid')
    corpus_sha=sha(CORPUS/'manifest.json')
    if corpus_sha!=(CORPUS/'manifest.sha256').read_text(encoding='ascii').strip():raise ValueError('Corpus changed')
    rows=json.loads((CORPUS/'manifest.json').read_text(encoding='utf-8'))['documents']
    out=ART/f'complex_ingestion_{args.mode}_v1';out.mkdir(exist_ok=True)
    records=[];loaded_by_id={}
    for row in rows:
        path=CORPUS/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Document changed')
        prediction_sha=None
        if args.mode=='enhanced' and row['kind']=='image_pdf':
            pred_path=ART/args.ocr_dir/'predictions'/(row['id']+'.json')
            pred=json.loads(pred_path.read_text(encoding='utf-8'))
            if pred['input_pdf_sha256']!=row['sha256'] or pred['status']!='ok':raise ValueError('OCR invalid or different input')
            sections=_split_markdown_sections(pred['text'])
            for section in sections:section.metadata.update(ocr_used=True,page_number=1)
            loaded=LoadedDocument(path.name,'pdf','dashscope_vision_ocr',sections)
            prediction_sha=sha(pred_path)
        else:loaded=load_document(path)
        loaded_by_id[row['id']]=loaded
        records.append({'id':row['id'],'file':row['file'],'input_sha256':row['sha256'],'kind':row['kind'],
                        'prediction_sha256':prediction_sha,'parser':loaded.parser_name,'section_count':len(loaded.sections),
                        'parsed_text':loaded.full_text,'parsed_quality':evaluate(row,loaded.full_text)})
    contract={'mode':args.mode,'corpus_sha256':corpus_sha,'rules':RULES,'denominator':20,'ocr_enabled_in_native_loader':False,
              'pipeline':'project ingest_loaded_document + project splitters + real embedding + PG/Milvus verification',
              'documents':[{'id':r['id'],'input_sha256':r['input_sha256'],'prediction_sha256':r['prediction_sha256']} for r in records]}
    cp=out/'contract.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8'))!=contract:raise ValueError('Existing run contract differs')
    write(cp,contract);write(out/'parsed.json',records)
    if args.prepare_only:
        print(json.dumps({'mode':args.mode,'parsed_usable':sum(r['parsed_quality']['usable_content'] for r in records),'files':20,'real_ingestion':False}));return
    original=get_settings();url=make_url(original.postgres_dsn)
    if url.host not in {'127.0.0.1','localhost'} or url.database!='rag_benchmark' or original.milvus_collection!='tidb_benchmark_v1':raise ValueError('Original environment differs')
    db_name=f'rag_complex_{args.mode}_v1';collection=f'complex_{args.mode}_v1'
    admin=create_engine(_normalize_sqlalchemy_dsn(url.set(database='postgres').render_as_string(hide_password=False)),isolation_level='AUTOCOMMIT')
    with admin.connect() as con:
        exists=con.execute(text('SELECT 1 FROM pg_database WHERE datname=:name'),{'name':db_name}).first()
        if not exists:con.exec_driver_sql(f'CREATE DATABASE "{db_name}"')
    admin.dispose()
    os.environ['POSTGRES_DSN']=url.set(database=db_name).render_as_string(hide_password=False)
    os.environ['MILVUS_COLLECTION']=collection
    get_settings.cache_clear();get_engine.cache_clear();get_session_factory.cache_clear()
    settings=get_settings();engine=get_engine();Base.metadata.create_all(engine)
    with Session(engine) as db:existing=db.execute(select(Document)).scalars().all()
    result_path=out/'results.json'
    if existing:
        if not result_path.exists():raise ValueError('Target has previous incomplete data; audit before resume')
        saved=json.loads(result_path.read_text(encoding='utf-8'))
        snap=snapshot_pg(engine);check_vectors(snap,settings)
        if snap!=json.loads((out/'snapshot.json').read_text(encoding='utf-8')):raise ValueError('Cached database changed')
        print(json.dumps({'cached':True,**saved['summary']},ensure_ascii=False));return
    client=MilvusClient(uri=settings.resolved_milvus_uri,timeout=15)
    try:
        if client.has_collection(collection):raise ValueError('Collection exists but database empty')
    finally:client.close()
    events=[];failures=[];labels={row['id']:row for row in rows}
    with ExitStack() as stack:
        embedding_client=get_embeddings().client
        stack.enter_context(patch.object(embedding_client,'call',side_effect=track_embedding_call(embedding_client.call,events)))
        for record in records:
            path=CORPUS/record['file']
            with get_session_factory()() as db:
                try:
                    doc=ingest_loaded_document(db,loaded_document=loaded_by_id[record['id']],knowledge_base='complex_document_benchmark',
                        source_path=str(path),file_size=path.stat().st_size)
                    db.commit()
                    chunks=db.execute(select(Chunk).where(Chunk.document_id==doc.id).order_by(Chunk.chunk_index)).scalars().all()
                    persisted='\n\n'.join(c.content for c in chunks)
                    record.update(document_id=doc.id,status=doc.status,chunk_count=len(chunks),persisted_text=persisted,
                        persisted_quality=evaluate(labels[record['id']],persisted),execution='ok')
                except Exception as exc:
                    db.rollback();record.update(execution='error',error_type=type(exc).__name__)
                    failures.append(record['id'])
            write(out/'checkpoint.json',records)
            print(json.dumps({'id':record['id'],'execution':record['execution'],'chunks':record.get('chunk_count',0),
                'usable':record.get('persisted_quality',{}).get('usable_content',False)},ensure_ascii=False),flush=True)
            if failures:break
    write(out/'embedding_events.json',events)
    if failures:raise RuntimeError('Ingestion failed; preserve checkpoint and audit before resume')
    snapshot=snapshot_pg(engine);check_vectors(snapshot,settings)
    write(out/'snapshot.json',snapshot)
    for record in records:
        record['pg_milvus_verified']=True
        record['usable']=bool(record['chunk_count']) and record['parsed_quality']['usable_content'] and record['persisted_quality']['usable_content']
    count=sum(r['usable'] for r in records)
    summary={'mode':args.mode,'denominator':20,'usable_files':count,'usable_rate':count/20,'api_errors':0,
             'pg_documents':len(snapshot['documents']),'pg_chunks':len(snapshot['chunks']),'pg_milvus_verified':True,
             'embedding_tokens':sum(e.get('usage',{}).get('total_tokens',0) for e in events)}
    write(result_path,{'contract_sha256':sha(cp),'records':records,'summary':summary})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
