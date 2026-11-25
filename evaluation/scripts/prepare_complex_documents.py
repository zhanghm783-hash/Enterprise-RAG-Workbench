"""模型执行前固定20份企业复杂文档：10图片型研报PDF＋10研报表格DOCX。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import subprocess

from docx import Document
from docx.shared import Pt
from lxml import html
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'evaluation/corpus/omnidocbench_full_v1'
DEST=ROOT/'evaluation/corpus/complex_enterprise_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def table_rows(markup):
    root=html.fromstring(markup)
    rows=root.xpath('.//tr') if root.tag!='tr' else [root]
    return [[{'text':' '.join(cell.itertext()).strip(),'rowspan':int(cell.get('rowspan','1')),'colspan':int(cell.get('colspan','1'))}
             for cell in row.xpath('./th|./td')] for row in rows]


def expanded(rows):
    grid={}
    for i,row in enumerate(rows):
        col=0
        for cell in row:
            while (i,col) in grid:col+=1
            for di in range(cell['rowspan']):
                for dj in range(cell['colspan']):grid[(i+di,col+dj)]=cell['text']
            col+=cell['colspan']
    height=max((i for i,j in grid),default=-1)+1;width=max((j for i,j in grid),default=-1)+1
    return [[grid.get((i,j),'') for j in range(width)] for i in range(height)]


def main():
    meta=json.loads((SOURCE/'source.json').read_text(encoding='utf-8'))
    if sha(SOURCE/'annotations.json')!=meta['sha256']:raise ValueError('Source annotations changed')
    if (DEST/'manifest.json').exists():print('Complex document corpus already frozen.');return
    candidates=[]
    for page in json.loads((SOURCE/'annotations.json').read_text(encoding='utf-8')):
        attr=page['page_info']['page_attribute']
        if attr.get('data_source')!='research_report' or attr.get('language')!='simplified_chinese' or attr.get('layout')!='single_column':continue
        tables=[b for b in page['layout_dets'] if b['category_type']=='table' and not b.get('ignore') and b.get('html')]
        text=[b['text'] for b in page['layout_dets'] if b['category_type'] in ('text_block','title','table_caption') and not b.get('ignore') and b.get('text')]
        if not 1<=len(tables)<=2 or not 150<=sum(map(len,text))<=3000:continue
        grids=[expanded(table_rows(t['html'])) for t in tables]
        if not all(2<=len(g)<=14 and 2<=max(map(len,g),default=0)<=8 for g in grids):continue
        cells=sum(sum(bool(v.strip()) for row in grid for v in row) for grid in grids)
        if not 12<=cells<=120:continue
        candidates.append((page,tables,text,grids))
    candidates.sort(key=lambda item:item[0]['page_info']['image_path'])
    if len(candidates)<20:raise ValueError('Not enough predefined eligible pages')
    selected=candidates[:20];DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'images').mkdir(exist_ok=True);(DEST/'documents').mkdir(exist_ok=True)
    labels=[]
    for index,(page,tables,text,grids) in enumerate(selected,1):
        name=page['page_info']['image_path'];pdf=index<=10
        entry={'id':f'C{index:02d}','kind':'image_pdf' if pdf else 'table_docx','source_image_path':name,
               'source_sample_id':page['page_info'].get('sample_id'),'source_page_no':page['page_info']['page_no'],
               'source_page_attributes':page['page_info']['page_attribute'],'reference_text_blocks':text if pdf else [],
               'reference_table_grids':grids if pdf else [grids[0]],'file':f'documents/{index:02d}_'+('scan.pdf' if pdf else 'table.docx')}
        labels.append(entry)
    # Labels and thresholds are written before any OCR or document quality measurement.
    plan={'status':'planned_before_ocr','source_revision':meta['revision'],'annotation_sha256':meta['sha256'],
          'selection':'first20 sorted eligible Chinese single-column research-report pages; 1-2 tables, 2-14 rows, 2-8 columns, 150-3000 text characters',
          'sample_design':'first10 image-only PDF wrappers of official page images; next10 DOCX fixtures recreated from distinct official table annotations',
          'thresholds':{'text_anchor_coverage_min':0.90,'table_cell_coverage_min':0.95,'table_row_association_min':0.90,
                        'persistence_required':'at least one real chunk, with PostgreSQL/Milvus text correspondence'},
          'research_only_dataset':True,'production_distribution':False,'documents':labels}
    (DEST/'pre_ocr_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def fetch(entry):
        name=entry['source_image_path'];target=DEST/'images'/name
        url=f'https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/{meta["revision"]}/images/'+urllib.parse.quote(name)
        if not target.exists():
            request=urllib.request.Request(url+'?download=true',headers={'User-Agent':'RAG-document-evaluation'})
            try:
                with urllib.request.urlopen(request,timeout=40) as response:target.write_bytes(response.read())
            except Exception as first_error:
                temporary=target.with_suffix(target.suffix+'.part')
                result=subprocess.run(['curl.exe','--location','--fail','--retry','2','--connect-timeout','15','--max-time','90',
                                       '--output',str(temporary),url+'?download=true'],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('Official image download failed: '+type(first_error).__name__)
                temporary.replace(target)
        with Image.open(target) as im:im.verify()
        entry['source_image_url']=url;entry['source_image_sha256']=sha(target)
        return target
    with ThreadPoolExecutor(max_workers=2) as pool:
        images=list(pool.map(fetch,labels[:10]))
    for index,entry in enumerate(labels):
        path=DEST/entry['file'];page,tables,text,grids=selected[index]
        if entry['kind']=='image_pdf':
            image=images[index];width,height=A4
            c=canvas.Canvas(str(path),pagesize=A4);c.drawImage(str(image),0,0,width=width,height=height);c.showPage();c.save()
            if ''.join(p.extract_text() or '' for p in PdfReader(path).pages).strip():raise ValueError('PDF must have no hidden text layer')
        else:
            document=Document();document.add_heading('企业研报表格摘录',level=1)
            document.add_paragraph('公开研报表格的DOCX兼容性样本；对应页号：'+str(entry['source_page_no']))
            raw=table_rows(tables[0]['html']);grid=grids[0]
            table=document.add_table(rows=len(grid),cols=max(map(len,grid)));table.style='Table Grid';occupied=set()
            for i,row in enumerate(raw):
                col=0
                for cell in row:
                    while (i,col) in occupied:col+=1
                    end_i=i+cell['rowspan']-1;end_j=col+cell['colspan']-1
                    target=table.cell(i,col)
                    if end_i!=i or end_j!=col:target=target.merge(table.cell(end_i,end_j))
                    target.text=cell['text']
                    for di in range(cell['rowspan']):
                        for dj in range(cell['colspan']):occupied.add((i+di,col+dj))
                    col+=cell['colspan']
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:run.font.size=Pt(9)
            document.save(path)
        entry['sha256']=sha(path);entry['bytes']=path.stat().st_size
        print(entry['id']+' '+entry['kind']+' prepared',flush=True)
    plan['status']='frozen_before_ocr';plan['documents']=labels
    path=DEST/'manifest.json';path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (DEST/'manifest.sha256').write_text(sha(path)+'\n',encoding='ascii')
    print(json.dumps({'status':'frozen','documents':20,'pdf':10,'docx':10,'sha256':sha(path)},ensure_ascii=False))


if __name__=='__main__':main()
