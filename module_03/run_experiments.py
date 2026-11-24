"""Воспроизводимые замеры занятия 6. Запуск: python run_experiments.py."""
import contextlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import corpus_loader
import ingestion
import vector_store

BASE = Path(__file__).parent
RESULTS = BASE / 'results'


def measure(client, model, label, size, overlap):
    ingestion.CHUNK_SIZE, ingestion.OVERLAP = size, overlap
    with (RESULTS / f'{label}.txt').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            manifest, chunks = ingestion.build_index(client, model)
            ingestion.show_sample(chunks)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            for doc_type in (None, 'регламент'):
                print(f'\nПОИСК: type={doc_type}')
                for query in vector_store.QUERIES:
                    hits = vector_store.search(client, model, query, ingestion.ALIAS, doc_type)
                    if doc_type:
                        assert all(h.payload['type'] == doc_type for h in hits)
                    vector_store.show(hits, query)
            docs, facts, total = vector_store.check_questions(client, model, ingestion.ALIAS)
            baseline, _ = vector_store.baseline_whole_documents(model)
            vector_store.threshold_report(client, model, ingestion.ALIAS)
            details = []
            for q in vector_store.QUESTIONS:
                hits = vector_store.search(client, model, q['question'], ingestion.ALIAS)
                details.append({'id': q['id'], 'question': q['question'],
                                'in_corpus': q['in_corpus'],
                                'best_score': hits[0].score if hits else 0,
                                'doc_ids': [h.payload['doc_id'] for h in hits],
                                'fact_found': bool(q['must_contain']) and any(
                                    q['must_contain'].lower() in h.payload['text'].lower() for h in hits)})
            # Проверяем сохранённый индекс, а не только список до загрузки.
            info = client.get_collection(ingestion.ALIAS)
            assert info.points_count == len(chunks)
            assert info.config.params.vectors.size == 384
            assert str(info.config.params.vectors.distance) == 'Cosine'
            pairs = [(a['payload'], b['payload']) for a,b in zip(chunks,chunks[1:])
                     if a['payload']['doc_id'] == b['payload']['doc_id']
                     and b['payload']['chunk_no'] == a['payload']['chunk_no']+1
                     and a['payload']['page'] == b['payload']['page']
                     and a['payload']['section'] == b['payload']['section']]
            if overlap:
                a,b = next((a,b) for a,b in pairs if len(a['text']) == size and len(b['text']) >= overlap)
                assert a['text'][-overlap:] == b['text'][:overlap]
                print('\nПроверенный нахлёст:', a['doc_id'], a['page'], a['section'])
                print(repr(a['text'][-overlap:]))
                print(repr(b['text'][:overlap]))
            print('\nКоллекции:', [c.name for c in client.get_collections().collections])
            print('Алиасы:', client.get_aliases())
    result = {'label': label, **manifest, 'document_hits': docs, 'fact_hits': facts,
              'total': total, 'baseline_hits': baseline, 'questions': details}
    print(label, len(chunks), f'doc={docs}/{total}', f'fact={facts}/{total}', flush=True)
    return result


def main():
    RESULTS.mkdir(exist_ok=True)
    model = SentenceTransformer(ingestion.MODEL_NAME, local_files_only=True, device='cpu')
    client = ingestion.make_client(False, ingestion.DB_PATH)
    original = corpus_loader.CORPUS_DIR
    results = []
    try:
        for size, overlap in [(600,150),(400,0),(1000,200),(300,150)]:
            results.append(measure(client, model, f'reference_{size}_{overlap}', size, overlap))
        corpus_loader.CORPUS_DIR = BASE / 'data' / 'corpus_my'
        results.append(measure(client, model, 'my_600_150', 600, 150))
    finally:
        corpus_loader.CORPUS_DIR = original
        # После экспериментов рабочий алиас снова указывает на эталон 600/150.
        ingestion.CHUNK_SIZE, ingestion.OVERLAP = 600, 150
        with (RESULTS / 'final_index.txt').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                ingestion.build_index(client, model)
        client.close()
    (RESULTS / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Проверка порога на контрольных вопросах\n']
    for result in results:
        lines.append(result['label'])
        for threshold in (0.0, 0.3, 0.5, 0.74):
            rejected = [q['id'] for q in result['questions']
                        if q['in_corpus'] and q['best_score'] < threshold]
            accepted = [q['id'] for q in result['questions']
                        if not q['in_corpus'] and q['best_score'] >= threshold]
            lines.append(f'MIN_SCORE={threshold}: отсечённые настоящие вопросы={rejected}; принятые ловушки={accepted}')
    (RESULTS / 'threshold_sweep.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')



if __name__ == '__main__':
    main()
