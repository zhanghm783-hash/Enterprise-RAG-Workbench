"""取得 BEIR 原始归档、核对官方 MD5，并清点评测规模；不调用模型。"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'evaluation/corpus/trec-covid'
URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip'
EXPECTED_MD5 = 'ce62140cb23feb9becf6270d0d1fe6d1'
FILES = {'corpus.jsonl': 'trec-covid/corpus.jsonl', 'queries.jsonl': 'trec-covid/queries.jsonl',
         'qrels/test.tsv': 'trec-covid/qrels/test.tsv'}


def file_hash(path, algorithm='sha256'):
    hasher = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive = DIRECTORY / 'trec-covid.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        size = 0
        next_progress = 16 * 1024 * 1024
        with urllib.request.urlopen(URL, timeout=45) as source, partial.open('wb') as output:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                output.write(block)
                size += len(block)
                if size >= next_progress:
                    print(f'Downloaded {size // (1024 * 1024)} MiB', flush=True)
                    next_progress += 16 * 1024 * 1024
                if size > 512 * 1024 * 1024:
                    raise ValueError('Unexpectedly large source archive')
        if file_hash(partial, 'md5') != EXPECTED_MD5:
            raise ValueError('Downloaded archive does not match official MD5')
        partial.replace(archive)
    if file_hash(archive, 'md5') != EXPECTED_MD5:
        raise ValueError('Existing archive does not match official MD5')
    with zipfile.ZipFile(archive) as zipped:
        for target_name, member in FILES.items():
            info = zipped.getinfo(member)
            if info.file_size > 1024 * 1024 * 1024:
                raise ValueError('Unexpectedly large archive member')
            target = DIRECTORY / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with zipped.open(info) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
            print(f'Prepared {target_name}: {target.stat().st_size} bytes', flush=True)

    queries = {}
    with (DIRECTORY / 'queries.jsonl').open(encoding='utf-8') as source:
        for line in source:
            query = json.loads(line)
            if query['_id'] in queries:
                raise ValueError('Duplicate query ID')
            queries[query['_id']] = query['text']
    documents, characters, empty, long_documents, token_estimate, max_characters = 0, 0, 0, 0, 0, 0
    all_ids = set()
    examples = []
    with (DIRECTORY / 'corpus.jsonl').open(encoding='utf-8') as source:
        for line in source:
            item = json.loads(line)
            if item['_id'] in all_ids:
                raise ValueError('Duplicate corpus ID')
            all_ids.add(item['_id'])
            text = '\n'.join(part.strip() for part in (item['title'], item['text']) if part.strip())
            length = len(text)
            documents += 1
            characters += length
            token_estimate += math.ceil(length / 4)
            empty += not bool(text)
            long_documents += length > 6000
            max_characters = max(max_characters, length)
            if len(examples) < 2:
                examples.append({'id': item['_id'], 'title': item['title'], 'characters': length})
    judgments, positives = Counter(), Counter()
    invalid = 0
    with (DIRECTORY / 'qrels/test.tsv').open(encoding='utf-8') as source:
        next(source)
        for line in source:
            qid, pid, grade = line.rstrip('\n').split('\t')
            grade = int(grade)
            invalid += qid not in queries or pid not in all_ids
            judgments[grade] += 1
            if grade > 0:
                positives[qid] += 1
    if invalid or set(queries) - set(positives):
        raise ValueError('Qrels references or answerable query coverage invalid')
    inventory = {
        'prepared_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source_url': URL, 'official_md5': EXPECTED_MD5, 'archive_sha256': file_hash(archive),
        'archive_bytes': archive.stat().st_size,
        'files': {name: {'bytes': (DIRECTORY / name).stat().st_size, 'sha256': file_hash(DIRECTORY / name)}
                  for name in FILES},
        'documents': documents, 'queries': len(queries), 'characters_title_plus_text': characters,
        'empty_title_and_text': empty, 'max_document_characters': max_characters,
        'documents_over_6000_characters': long_documents,
        'qrels_rows': sum(judgments.values()), 'grades': dict(judgments),
        'min_positive_documents_per_query': min(positives.values()),
        'max_positive_documents_per_query': max(positives.values()),
        'rough_english_tokens_character_div_four': token_estimate,
        'rough_full_embedding_cost_rmb_at_point7_per_million': token_estimate * 0.7 / 1_000_000,
        'estimate_note': 'Character/4 approximation, not provider tokenization or bill; no truncation/splitting policy applied yet',
        'document_examples': examples, 'first_queries': list(queries.items())[:5],
        'paid_model_calls': 0,
        'evaluation_notes': [
            'Original task has graded document relevance; nDCG@10 is needed alongside Hit Rate@5',
            'Some abstracts exceed embedding input limit; establish truncation or chunk-to-document aggregation before indexing',
            'Paper-level qrels do not label every chunk relevant; keep document-level evaluation separate from TiDB chunk-level benchmark',
            'Preserve original query text and all qrels, including grade 0 and unjudged-document distinction',
        ],
    }
    (DIRECTORY / 'source_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(inventory, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
