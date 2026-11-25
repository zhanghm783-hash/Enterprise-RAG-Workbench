"""离线核对公开逐题结果的计数与均值；不调用模型或服务。"""
import json,math
from pathlib import Path
from statistics import mean
P=Path(__file__).resolve().parent/'results'
def load(name):return json.loads((P/name).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
r=load('retrieval.json');assert len(r['rows'])==r['questions']==24
for mode in ['vector','bm25','rrf','improved']:
    count=sum(row[mode] for row in r['rows']);assert count==r['totals'][mode];close(count/24,r['rates'][mode])
for name,field in [('faithfulness.json','paired_rows'),('relevancy.json','rows')]:
    r=load(name);rows=r[field];assert len(rows)==24 and len({v['question_id'] for v in rows})==24
    for side in ['before','after']:close(mean(v[side] for v in rows),r[side])
r=load('ingestion.json')
for mode in ['native','enhanced']:
    d=r['data'][mode];assert len(d['records'])==d['total']==20
    count=sum(v['usable'] for v in d['records']);assert count==d['usable'];close(count/20,d['rate'])
print('PASS: retrieval counts, two paired answer-score means and ingestion counts match public reports.')
