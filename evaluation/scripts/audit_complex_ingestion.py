"""离线复算入库质量；区分逐字标注匹配与显示形式等价匹配。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from complex_document_quality import evaluate,table_grids,row_signature,normalize,RULES

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'evaluation/artifacts'
CORPUS=ROOT/'evaluation/corpus/complex_enterprise_v1'


def canonical(value):
    value=normalize(value).casefold().replace('$','').replace('\\pm','±').replace('\\sim','~').replace('～','~')
    value=value.replace('−','-').replace('–','-')
    # 标点形式等价，不删除数字、汉字、字母或单位；数字附近的正负号、百分号和小数保留。
    result=[]
    for i,ch in enumerate(value):
        if unicodedata.category(ch)[0] in {'L','N'} or ch in '%±~':result.append(ch)
        elif ch in '+-.／/':
            left=value[i-1:i];right=value[i+1:i+2]
            if left.isdigit() or right.isdigit():result.append(ch)
    return ''.join(result)


def equivalent_quality(label,body):
    base=evaluate(label,body)
    _,grids=table_grids(body)
    actual_rows=[[canonical(cell) for cell in row] for grid in grids for row in grid]
    actual_rows=[[cell for cell in row if cell] for row in actual_rows]
    # 合并单元格在不同HTML中可能分成相邻两格；只允许同一行的连续文字匹配，不跨行拼接。
    expected_rows=[row_signature(row) for grid in label['reference_table_grids'] for row in grid]
    expected_rows=[[canonical(cell) for cell in row if canonical(cell)] for row in expected_rows]
    cells={cell for row in expected_rows for cell in row}
    def matches(cell,row,start=0):
        numeric=bool(re.fullmatch(r'[+\-±]?\d+(?:\.\d+)?%?',cell))
        for i in range(start,len(row)):
            for end in range(i+1,min(i+2,len(row)+1) if numeric else len(row)+1):
                if ''.join(row[i:end])==cell:yield end
    def contained(cell):return any(any(matches(cell,row)) for row in actual_rows)
    def ordered(row,actual):
        cursors={0}
        for cell in row:
            cursors={end for cursor in cursors for end in matches(cell,actual,cursor)}
            if not cursors:return False
        return bool(cursors)
    row_labels=[row for row in expected_rows if len(row)>=2]
    missing=sorted(cell for cell in cells if not contained(cell))
    cell_rate=(len(cells)-len(missing))/len(cells) if cells else 1
    row_hits=sum(any(ordered(row,actual) for actual in actual_rows) for row in row_labels)
    row_rate=row_hits/len(row_labels) if row_labels else 1
    return {**base,'table_cell_coverage':cell_rate,'missing_cells':missing,'table_row_association':row_rate,
        'matched_rows':row_hits,'reference_rows':len(row_labels),
        'usable_content':bool(body.strip()) and base['text_anchor_coverage']>=.9 and cell_rate>=.95 and row_rate>=.9}


def main():
    label={'reference_text_blocks':[],'reference_table_grids':[[['项目','金额'],['项目甲','100'],['项目乙','200']]]}
    good='<table><tr><td>项目</td><td>金额</td></tr><tr><td>项目甲</td><td>100</td></tr><tr><td>项目乙</td><td>200</td></tr></table>'
    assert equivalent_quality(label,good)['usable_content']
    assert not equivalent_quality(label,good.replace('100','1000'))['usable_content']
    assert not equivalent_quality(label,good.replace('100','TEMP').replace('200','100').replace('TEMP','200'))['usable_content']
    assert not equivalent_quality(label,good.replace('<td>100</td>','<td>10</td><td>0</td>'))['usable_content']
    assert canonical(r'5%$\sim$20%')==canonical('5%～20%')
    assert canonical('-5%')!=canonical('5%') and canonical('6万元')!=canonical('6亿元')
    assert not equivalent_quality(label,'')['usable_content']
    corpus_sha=hashlib.sha256((CORPUS/'manifest.json').read_bytes()).hexdigest()
    assert corpus_sha==(CORPUS/'manifest.sha256').read_text(encoding='ascii').strip()
    label_doc=json.loads((CORPUS/'manifest.json').read_text(encoding='utf-8'))
    labels={row['id']:row for row in label_doc['documents']}
    assert len(labels)==20
    for row in labels.values():assert hashlib.sha256((CORPUS/row['file']).read_bytes()).hexdigest()==row['sha256']
    outputs={}
    for mode in ['native','enhanced']:
        path=ART/f'complex_ingestion_{mode}_v1/results.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        if len(data['records'])!=20 or not data['summary']['pg_milvus_verified']:raise ValueError('Incomplete persistence')
        assert {r['id'] for r in data['records']}==set(labels)
        records=[]
        for row in data['records']:
            label=labels[row['id']]
            if row['input_sha256']!=label['sha256'] or not row['pg_milvus_verified']:raise ValueError('Input/persistence mismatch')
            parsed=equivalent_quality(label,row['parsed_text']);stored=equivalent_quality(label,row['persisted_text'])
            records.append({'id':row['id'],'parsed_quality':parsed,'persisted_quality':stored,
                'usable':bool(row['chunk_count']) and parsed['usable_content'] and stored['usable_content'],
                'strict_v1_usable':row['usable']})
        usable=sum(r['usable'] for r in records)
        outputs[mode]={'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'usable':usable,'total':20,'rate':usable/20,'strict_v1_rate':data['summary']['usable_rate'],'records':records}
    report={'scope':label_doc['sample_design'],'post_first_ocr_audit_revision':True,'corpus_sha256':corpus_sha,
        'validator_safety_checks':'8 passed: values, numeric boundary, row association, split numeric, math, sign, unit, empty',
        'rule_version':'display_equivalent_v2','rule_change':'Canonicalize LaTeX math, case and punctuation; allow adjacent cells in same row; exact characters/numbers/units remain required; no fuzzy edits or gold correction.',
        'thresholds':RULES,'both_versions_rescored_identically':True,'raw_strict_v1_results_preserved':True,
        'data':outputs,'delta':outputs['enhanced']['rate']-outputs['native']['rate']}
    target=ART/'complex_document_comparison_v2.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({m:{k:v for k,v in r.items() if k in ['usable','total','rate','strict_v1_rate']} for m,r in outputs.items()},ensure_ascii=False))


if __name__=='__main__':main()
