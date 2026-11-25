"""准备隔离入库的可核对计划；可选只读查重，不创建库、不向量化。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.config import get_settings  # noqa: E402
from app.rag.ingest import _split_section  # noqa: E402
from app.rag.loader import load_document  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-services', action='store_true', help='Only read local PG catalog and Milvus collection names')
    args = parser.parse_args()
    settings = get_settings()
    active_database = make_url(settings.postgres_dsn).database
    target_database, target_collection = 'rag_enterprise_v2', 'tidb_enterprise_v2'
    if active_database == target_database or settings.milvus_collection == target_collection:
        raise ValueError('Planned isolated target must differ from active configuration')
    if settings.embedding_model != 'text-embedding-v1' or settings.milvus_dimension != 1536:
        raise ValueError('Model/dimension changed; update price and workload plan explicitly')
    corpus = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
    manifest_bytes = (corpus/'source_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    chunk_rows, documents = [], []
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline preparation network call')):
        for item in manifest['files']:
            data = (corpus/item['local_path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError(f"Source changed: {item['local_path']}")
            if item['category'] == '出处与许可':
                continue
            document = load_document(corpus/item['local_path'])
            items = []
            for section in document.sections:
                for chunk in _split_section(text=section.text,file_type=document.file_type,section_metadata=section.metadata)[1]:
                    items.append({'corpus_path':item['local_path'], 'chunk_index':len(items),
                                  'content_sha256':hashlib.sha256(chunk.content.encode('utf-8')).hexdigest(),
                                  'characters':len(chunk.content), 'utf8_bytes':len(chunk.content.encode('utf-8'))})
            chunk_rows.extend(items)
            documents.append({'corpus_path':item['local_path'], 'upload_filename':document.filename,
                              'source_sha256':item['sha256'], 'expected_chunks':len(items),
                              'nominal_sdk_batches_at_25':math.ceil(len(items)/25)})
    if len({d['upload_filename'] for d in documents}) != len(documents):
        raise ValueError('Source basenames collide; preserve paths before ingestion')
    total_chars = sum(row['characters'] for row in chunk_rows)
    price = 0.7
    contract_path = ROOT/'evaluation/datasets/enterprise_tidb_v2_review2/scoring_contract_v2.json'
    plan = {
        'status':'prepared_not_executed', 'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        'isolation':{'configured_current_database':active_database,'configured_current_collection':settings.milvus_collection,
                     'planned_database':target_database,'planned_collection':target_collection,
                     'method':'单独进程在导入 app 模块前覆盖环境变量；复用本地服务，不改 backend/.env 或当前网页后端进程',
                     'existing_data_policy':'发现目标已有内容就停止检查，不自动删除或覆盖；保留旧库与旧集合'},
        'embedding':{'model':settings.embedding_model,'dimensions':1536,'price_rmb_per_million_tokens':price,
                     'price_source':'https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api',
                     'pricing_checked_date':'2026-11-23','batch_size':25,'max_tokens_per_row':2048,
                     'nominal_sdk_batches':sum(d['nominal_sdk_batches_at_25'] for d in documents),
                     'batch_note':'按项目当前按文档 add_texts、已安装 LangChain 每批25行计算；重试可能重复批次，非实际 HTTP 次数'},
        'workload':{'documents':len(documents),'expected_chunks':len(chunk_rows),'total_chunk_characters':total_chars,
                    'total_chunk_utf8_bytes':sum(row['utf8_bytes'] for row in chunk_rows),
                    'maximum_chunk_characters':max(row['characters'] for row in chunk_rows),
                    'maximum_chunk_utf8_bytes':max(row['utf8_bytes'] for row in chunk_rows),
                    'provider_row_token_limit_check':'pending_before_embedding; character count is not token count',
                    'chunks_above_2048_characters':[{'corpus_path':row['corpus_path'],'chunk_index':row['chunk_index'],
                                                    'characters':row['characters']} for row in chunk_rows if row['characters']>2048]},
        'cost_scenarios':{'assumption':'仅按总字符数乘1或2 Token/字符估量工作量，不是官方分词器或硬性费用上限；实际使用响应 usage 与账单核对',
                          'one_token_per_character_rmb':round(total_chars/1e6*price,6),
                          'two_tokens_per_character_rmb':round(total_chars*2/1e6*price,6),
                          'not_included':'查询向量化、聊天生成、失败重试、此前操作；没有预扣款或调用模型'},
        'documents':documents,'planned_chunks':chunk_rows,'database_created':False,'index_created':False,
        'paid_model_calls':0,
    }
    destination=ROOT/'evaluation/artifacts/enterprise_ingestion_plan_v2'
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.check_services:
        from sqlalchemy import func, select, text
        from app.core.postgres import get_engine
        from app.core.milvus_client import get_milvus_client
        from app.models.document import Document
        from app.models.chunk import Chunk
        check={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'read-only target-name check; no creation, indexing or model call'}
        try:
            with get_engine().connect() as connection:
                check['target_database_exists']=bool(connection.execute(text('SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = :name)'),{'name':target_database}).scalar())
                check['configured_database']=connection.execute(text('SELECT current_database()')).scalar()
                check['configured_document_count']=connection.execute(select(func.count()).select_from(Document)).scalar()
                check['configured_chunk_count']=connection.execute(select(func.count()).select_from(Chunk)).scalar()
            client=get_milvus_client()
            names=client.list_collections()
            check['target_collection_exists']=target_collection in names
            check['configured_collection_exists']=settings.milvus_collection in names
            check['safe_to_create_fresh_targets']=not check['target_database_exists'] and not check['target_collection_exists']
        except Exception as exc:
            check['check_error_type']=type(exc).__name__
            check['safe_to_create_fresh_targets']=False
        (destination/'service_name_check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(check,ensure_ascii=False,indent=2))
    print(json.dumps({key:plan[key] for key in ('status','isolation','embedding','workload','cost_scenarios','paid_model_calls')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
