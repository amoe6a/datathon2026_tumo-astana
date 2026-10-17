from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from ml.rag import RuleRetriever
from ml.qwen_rag import validate_decisions
from ml.scene import readable
from ml.video import Frame

ROOT=Path(__file__).resolve().parents[1]


def test_retrieval_uses_rules_outside_original_two_categories():
    index=RuleRetriever(ROOT/'rules/rag')
    assert len(index.cards)==29
    assert index.retrieve('кассир использует телефон',1)[0]['rule_id']=='CASH-003'
    assert 'CASH-003' in [c['rule_id'] for c in index.retrieve('cashier holds a smartphone',4)]
    assert 'ROOM-001' in [c['rule_id'] for c in index.retrieve('the door is open',4)]


def decision(rule_id='CASH-001',**changes):
    event=dict(rule_id=rule_id,timecode_sec=2.,observation='A visible synthetic actor leaves.',
               all_conditions_visible=True,exception_applies=False)
    event.update(changes)
    return {'events':[event],'uncertain':False}


@pytest.mark.parametrize('change',[{'all_conditions_visible':False},{'all_conditions_visible':'true'},
                                  {'exception_applies':True},{'exception_applies':None}])
def test_missing_conditions_and_exceptions_cannot_create_findings(change):
    card=RuleRetriever(ROOT/'rules/rag').by_id['CASH-001']
    events,uncertain=validate_decisions(decision(**change),[card],[0.,2.,3.])
    assert events==[] and uncertain


def test_review_only_rule_never_promoted_to_violation():
    card=RuleRetriever(ROOT/'rules/rag').by_id['ROOM-001']
    events,uncertain=validate_decisions(decision('ROOM-001'),[card],[0.,2.,3.])
    assert events==[] and uncertain


@pytest.mark.parametrize('change',[{'rule_id':'invented'},{'timecode_sec':200.},{'timecode_sec':True},
                                  {'timecode_sec':float('nan')}])
def test_ungrounded_decisions_fail(change):
    card=RuleRetriever(ROOT/'rules/rag').by_id['CASH-001']
    with pytest.raises(ValueError):validate_decisions(decision(**change),[card],[0.,2.,3.])


def test_blank_input_is_not_treated_as_clean_video():
    assert not readable([])
    assert not readable([Frame(0,np.zeros((64,64,3),dtype=np.uint8))])
    assert not readable([Frame(0,np.full((64,64,3),255,dtype=np.uint8))])
    image=np.zeros((64,64,3),dtype=np.uint8);image[:32]=180
    assert readable([Frame(0,image)])


def test_missing_uncertainty_is_conservative_but_missing_events_fails():
    events,uncertain=validate_decisions({'events':[]},[],[0.])
    assert events==[] and uncertain is True
    with pytest.raises(ValueError):validate_decisions({'uncertain':False},[],[0.])


def test_source_links_and_modes_preserved():
    index=RuleRetriever(ROOT/'rules/rag');catalog=index.catalog()
    assert catalog['ROOM-001']['decision_mode']=='review_only'
    assert catalog['CASH-001']['url']=='https://adilet.zan.kz/rus/docs/V1900019680'
    assert '22' in catalog['CASH-001']['section']


def test_guided_schema_limits_time_to_available_frames():
    from ml.qwen_rag import decision_schema
    card=RuleRetriever(ROOT/'rules/rag').by_id['CASH-001']
    props=decision_schema([card],[0.,1.5,1.5,3.])['properties']['events']['items']['properties']
    assert props['timecode_sec']['enum']==[0.,1.5,3.]
    assert props['rule_id']['enum']==['CASH-001']


def test_english_prompts_preserve_all_condition_exception_and_caution_entries():
    index=RuleRetriever(ROOT/'rules/rag')
    for card in index.cards:
        translated=index.prompt_card(card,'en')
        assert translated['mode']==card['decision_mode']
        for key,source in [('required_evidence','required_evidence'),('exceptions','exceptions_and_negative_examples'),('do_not_infer','do_not_infer')]:
            assert len(translated[key])==len(card[source])
    assert len(index.translations)==15


def test_changed_rules_require_translation_review(tmp_path):
    import json,shutil
    for name in ['rules.json','sources.json','conditional.en.json']:
        shutil.copyfile(ROOT/'rules/rag'/name,tmp_path/name)
    cards=json.loads((tmp_path/'rules.json').read_text(encoding='utf-8'))
    cards[0]['required_evidence'].append('A newly reviewed prerequisite.')
    (tmp_path/'rules.json').write_text(json.dumps(cards),encoding='utf-8')
    index=RuleRetriever(tmp_path)
    with pytest.raises(ValueError,match='out of date'):index.prompt_card(index.by_id['CASH-001'],'en')


def test_rag_events_keep_jury_schema_and_real_evidence(tmp_path,monkeypatch):
    import base64
    import json
    import cv2
    from ml.engine import analyze
    from scripts.make_demo import clip
    video=tmp_path/'synthetic.mov';clip(video,[],duration=3)
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'rag_folder':str(ROOT/'rules/rag')}))
    monkeypatch.setattr('ml.qwen_rag.analyze_rag',lambda *args:([
        {'rule_id':'CASH-003','timecode_sec':1.,'observation':'Synthetic integration event for contract testing.'}],[]))
    result=analyze(video,config,'qwen_rag',ROOT/'rules/catalog.json')
    assert result['verdict']=='violation' and result['meta']['engine']=='qwen_rag'
    event=result['violations'][0]
    assert event['rule']['id']=='CASH-003' and event['rule']['section']=='Пункт 12'
    assert len(event['evidence_frames'])==3
    assert event['timecode_sec'] in [f['timestamp_sec'] for f in event['evidence_frames']]
    for frame in event['evidence_frames']:
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(frame['image_base64']),np.uint8),cv2.IMREAD_COLOR)
        assert pixels is not None
