"""固定正文/表格可用性判据；不以非空文本冒充文档可用。"""
from __future__ import annotations

import html as html_std
import re
import unicodedata

from lxml import html
import markdown

RULES={'version':'complex_document_quality_v1','text_anchor_characters':4,'text_anchor_stride':4,
       'text_anchor_coverage_min':0.90,'table_cell_coverage_min':0.95,'table_row_association_min':0.90,
       'normalization':'NFKC, decode HTML entities, ignore whitespace/backticks/thousands commas; retain decimals/signs/percent/units',
       'required_persistence':'at least one real chunk, same required content after splitting, PG/Milvus content correspondence'}


def normalize(value):
    value=unicodedata.normalize('NFKC',html_std.unescape(str(value)))
    return re.sub(r'[\s,，`]+','',value)


def table_grids(text):
    stripped=text.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        stripped=stripped.split('\n',1)[1].rsplit('```',1)[0]
    rendered=markdown.markdown(stripped,extensions=['tables'])
    root=html.fragment_fromstring(rendered,create_parent='div')
    result=[]
    for table in root.xpath('.//table'):
        cells={}
        for i,row in enumerate(table.xpath('./tr|./thead/tr|./tbody/tr|./tfoot/tr')):
            col=0
            for cell in row.xpath('./td|./th'):
                while (i,col) in cells:col+=1
                text_value=' '.join(cell.itertext()).strip()
                rs=max(1,int(cell.get('rowspan','1')));cs=max(1,int(cell.get('colspan','1')))
                for di in range(rs):
                    for dj in range(cs):cells[(i+di,col+dj)]=text_value
                col+=cs
        height=max((i for i,j in cells),default=-1)+1;width=max((j for i,j in cells),default=-1)+1
        result.append([[cells.get((i,j),'') for j in range(width)] for i in range(height)])
    return root.text_content(),result


def meaningful(value):return bool(value) and value not in {'-','--','—','/','N/A','NA'}


def row_signature(row):
    result=[]
    for cell in row:
        value=normalize(cell)
        if meaningful(value) and (not result or result[-1]!=value):result.append(value)
    return result


def ordered_subset(expected,actual):
    cursor=0
    for value in expected:
        while cursor<len(actual) and actual[cursor]!=value:cursor+=1
        if cursor==len(actual):return False
        cursor+=1
    return True


def evaluate(label,text):
    plain,predicted=table_grids(text)
    normalized=normalize(plain)
    anchors=[]
    for block in label.get('reference_text_blocks',[]):
        block=normalize(block)
        anchors.extend(block[i:i+4] for i in range(0,len(block)-3,4))
    anchor_hits=sum(anchor in normalized for anchor in anchors)
    text_rate=anchor_hits/len(anchors) if anchors else 1.0
    gold_cells={normalize(v) for grid in label['reference_table_grids'] for row in grid for v in row if meaningful(normalize(v))}
    actual_cells={normalize(v) for grid in predicted for row in grid for v in row if meaningful(normalize(v))}
    missing=sorted(gold_cells-actual_cells)
    cell_rate=len(gold_cells&actual_cells)/len(gold_cells) if gold_cells else 1.0
    expected_rows=[row_signature(row) for grid in label['reference_table_grids'] for row in grid if len(row_signature(row))>=2]
    actual_rows=[row_signature(row) for grid in predicted for row in grid]
    row_hits=sum(any(ordered_subset(expected,actual) for actual in actual_rows) for expected in expected_rows)
    row_rate=row_hits/len(expected_rows) if expected_rows else 1.0
    return {'usable_content':bool(text.strip()) and text_rate>=RULES['text_anchor_coverage_min']
                and cell_rate>=RULES['table_cell_coverage_min'] and row_rate>=RULES['table_row_association_min'],
            'text_anchor_coverage':text_rate,'text_anchors':len(anchors),'text_anchor_hits':anchor_hits,
            'table_cell_coverage':cell_rate,'reference_unique_cells':len(gold_cells),'missing_cells':missing,
            'table_row_association':row_rate,'reference_rows':len(expected_rows),'matched_rows':row_hits,
            'predicted_tables':len(predicted)}


if __name__=='__main__':
    label={'reference_text_blocks':[],'reference_table_grids':[[['项目','金额'],['项目甲','100'],['项目乙','200']]]}
    good='<table><tr><td>项目</td><td>金额</td></tr><tr><td>项目甲</td><td>100</td></tr><tr><td>项目乙</td><td>200</td></tr></table>'
    wrong=good.replace('<td>100</td>','<td>TEMP</td>').replace('<td>200</td>','<td>100</td>').replace('<td>TEMP</td>','<td>200</td>')
    assert evaluate(label,good)['usable_content']
    assert evaluate(label,wrong)['table_cell_coverage']==1 and not evaluate(label,wrong)['usable_content']
    assert not evaluate(label,'')['usable_content']
    assert not evaluate(label,'项目 金额 项目甲100 项目乙200')['usable_content']
    assert normalize('1,234.50%')==normalize('1234.50%')
    assert normalize('-1234.50%')!=normalize('1234.50%')
    print('6 document-quality checks passed: values, row association, empty/flattened rejection and numeric signs.')
