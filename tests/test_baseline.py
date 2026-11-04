import json
from pathlib import Path
from src.catalog import ICDCatalog
from src.clinical import get_diagnosis_focus
from src.main import run

ROOT = Path(__file__).resolve().parents[1]

def test_catalog_count_and_known_code():
    c = ICDCatalog(ROOT/'data/mms-2026-01.json')
    assert len(c.records) == 18375
    assert c.is_valid('1F40.Z')
    assert not c.is_valid('FAKE.CODE')
    assert c.sanitize(['FAKE.CODE', '1F40.Z', '1F40.Z']) == ['1F40.Z']

def test_diagnosis_extraction():
    assert 'typhoid' in get_diagnosis_focus('S: Fever\nImp: Enteric fever - typhoid.\nPlan: treatment')

def test_file_runner_no_gold_needed(tmp_path):
    notes = tmp_path/'notes.jsonl'
    out = tmp_path/'predictions.jsonl'
    notes.write_text('\n'.join([
        json.dumps({'id':'x','text':'Dx: Confirmed pulmonary tuberculosis'}),
        json.dumps({'id':'y','text':'Dx: Chickenpox'}),
        json.dumps({'id':'z','text':''}),
    ])+'\n', encoding='utf-8')
    assert run(str(notes),str(out))==3
    rows=[json.loads(x) for x in out.read_text().splitlines()]
    assert [x['id'] for x in rows]==['x','y','z']
    catalog=ICDCatalog(ROOT/'data/mms-2026-01.json')
    assert all(1 <= len(r['codes']) <= 3 for r in rows)
    assert all(catalog.is_valid(code) for r in rows for code in r['codes'])
