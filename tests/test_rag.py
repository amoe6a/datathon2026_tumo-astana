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
