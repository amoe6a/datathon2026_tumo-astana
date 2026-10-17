"""Run local synthetic GPU generation while rejecting every socket connection."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')


def main(args):
    attempts=[]
    def blocked(*_args,**_kwargs):
        attempts.append(True)
        raise RuntimeError('Network connection prohibited during offline check')
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    import numpy as np
    from ml.video import Frame
    from ml.qwen_rag import RagSession,load_config,decision_schema,validate_decisions
    from ml.scene import read_json
    before=time.monotonic();session=RagSession(load_config(args.config))
    frames=[Frame(t,np.zeros((64,64,3),dtype=np.uint8)) for t in [0.,1.,2.]]
    value=read_json(session.generate(frames,'Return only JSON {"ok":true}.',32,adapter=False,
        schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}))
    assert value=={'ok':True}
    cards=[session.retriever.by_id['CASH-001']]
    multilingual=read_json(session.generate(frames,
        'Synthetic format test, not real footage. Return exactly one event with rule_id CASH-001, '
        'timecode_sec 1.0, observation "Синтетическая сцена для проверки формата ответа.", '
        'all_conditions_visible true, exception_applies false and top-level uncertain false.',512,
        schema=decision_schema(cards,[0.,1.,2.])))
    events,_=validate_decisions(multilingual,cards,[0.,1.,2.])
    assert len(events)==1 and any('\u0400'<=c<='\u052f' for c in events[0]['observation'])
    # Exercise observation, local retrieval and visual verification, not only load.
    pattern=np.zeros((96,128,3),dtype=np.uint8);pattern[:48]=160
    pipeline_events,pipeline_warnings=session.analyze([Frame(t,pattern.copy()) for t in [0.,1.,2.]])
    assert isinstance(pipeline_events,list) and isinstance(pipeline_warnings,list)
    assert len(attempts)==0
    result={'passed':True,'network_connect_attempts':len(attempts),'synthetic_only':True,'cyrillic_event_schema_passed':True,
            'full_synthetic_pipeline_passed':True,
            'seconds':round(time.monotonic()-before,3),'scope':'Model load, tokenizer, constrained GPU generation and complete synthetic analysis pipeline'}
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
