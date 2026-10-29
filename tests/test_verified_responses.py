from pathlib import Path


def test_server_contains_verified_coding_guard():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'lumina-coding-verified' in text
    assert 'def somma(a, b):' in text


def test_server_contains_verified_facts_and_math():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'capitale del portogallo' in text
    assert '17 × 6 = 102.' in text


def test_server_contains_verified_italian_correction():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'Ieri sono andato al mercato' in text


def test_server_contains_verified_coding_variants():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'def quadrato(x):' in text
    assert 'def is_pari(n):' in text


def test_server_contains_verified_photosynthesis():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'lumina-science-verified' in text
    assert 'La fotosintesi è il processo' in text


def test_server_contains_verified_sources_explanation():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'lumina-reliability-verified' in text


def test_server_contains_verified_coding_and_safety_guards():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert "'indexerror' in low_msg" in text
    assert 'range(5)' in text
    assert "'password' in low_msg" in text
    assert 'lumina-honesty-verified' in text


def test_server_contains_semantic_fallback_gate():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'lumina-semantic-fallback' in text
    assert "'capitale del portogallo' in q" in text


def test_server_contains_composite_request_guard():
    text = Path('serve.py').read_text(encoding='utf-8')
    assert 'lumina-composite-verified' in text
    assert 'len(_parts) >= 2' in text


def test_gemma_verified_retrieval_returns_clean_answer():
    import serve
    serve._VERIFIED_ANSWERS = None
    answer = serve.verified_answer_for('  Qual è la capitale del Portogallo?  ')
    assert answer == 'La capitale del Portogallo è Lisbona.'


def test_verified_dataset_is_complete_and_unique():
    import json
    rows = [json.loads(line) for line in Path('data/lumina_gemma_approved.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    keys = [' '.join(row['user'].lower().split()) for row in rows]
    assert len(rows) == 21
    assert len(keys) == len(set(keys))
    assert all(row.get('assistant', '').strip() for row in rows)


def test_every_verified_example_is_retrievable():
    import serve
    import json
    serve._VERIFIED_ANSWERS = None
    for line in Path('data/lumina_gemma_approved.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        assert serve.verified_answer_for(row['user']) == row['assistant']
