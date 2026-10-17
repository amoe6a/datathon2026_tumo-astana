"""Local supervised QLoRA on human annotations. Logs contain aggregate metrics only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from ml.scene import load_base, visual_content, select_frames, SCENE_PROMPT
    from ml.video import sample_video
    random.seed(args.seed);torch.manual_seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    train_rows=[r for r in manifest['videos'] if r['split']=='train']
    if not train_rows:
        raise ValueError('No training videos')
    start=time.monotonic()
    report={'status':'loading','seed':args.seed,'training_videos':len(train_rows),
        'epochs_requested':args.epochs,'objective':'human annotation supervised scene description',
        'base_model':str(Path(args.model).resolve()),'quantization':'NF4 double quantization',
        'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'rank':8,'learning_rate':args.learning_rate,'gradient_accumulation':4,
        'vision_encoder_frozen':True,'steps':[],'checkpoints':[]}
    report['source_sha256']={str(p.relative_to(Path(__file__).resolve().parents[1])):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'ml/scene.py']}
    def save():
        report['elapsed_sec']=round(time.monotonic()-start,3)
        report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        temp=output/'status.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(output/'status.json')
    save()
    model,processor=load_base(args.model,quantized=True,training=True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    targets=[name for name,_ in model.named_modules()
             if 'language_model.layers.' in name and name.endswith(('.q_proj','.v_proj'))]
    if not targets:raise ValueError('Could not locate language-only adapter targets')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
        target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    trainable=[p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n):
        raise ValueError('Vision encoder must stay frozen')
    report['trainable_parameters']=sum(p.numel() for p in trainable)
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=0.01)
    cache=[]
    for row in train_rows:
        frames,duration=sample_video(row['path'],fps=1)
        selected=select_frames(frames,args.frames,row['anchors'])
        # Keep complete manually supplied annotations, without generating pseudo-ground-truth.
        answer='\n'.join(row['annotations'])
        messages=[{'role':'user','content':visual_content(selected,SCENE_PROMPT,side=384)}]
        prompt=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt')
        full=processor.apply_chat_template(messages+[{'role':'assistant','content':answer}],
            tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt')
        n=prompt['input_ids'].shape[1]
        if not torch.equal(prompt['input_ids'],full['input_ids'][:,:n]):
            raise ValueError('Assistant loss mask does not align with chat template')
        if full['input_ids'].shape[1]>6144:raise ValueError('Example exceeds bounded token budget')
        labels=full['input_ids'].clone();labels[:,:n]=-100
        if not torch.any(labels!=-100):raise ValueError('No supervised answer tokens')
        full['labels']=labels;cache.append(full)
    report['max_input_tokens']=max(x['input_ids'].shape[1] for x in cache)
    report['status']='training';save()
    model.train();optimizer.zero_grad(set_to_none=True);micro=0
    for epoch in range(args.epochs):
        order=list(range(len(cache)));random.shuffle(order);losses=[]
        for offset,index in enumerate(order):
            batch={k:v.to('cuda') if hasattr(v,'to') else v for k,v in cache[index].items()}
            before=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(**batch,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            group_start=(offset//4)*4
            group_size=min(4,len(order)-group_start)
            if args.max_microsteps:
                group_size=min(group_size,args.max_microsteps-(micro-offset%4))
            value=float(loss.detach());(loss/group_size).backward();micro+=1;losses.append(value)
            last=offset==len(order)-1 or (args.max_microsteps and micro>=args.max_microsteps)
            if (offset+1)%4==0 or last:
                torch.nn.utils.clip_grad_norm_(trainable,1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
            report['steps'].append({'microstep':micro,'epoch':epoch+1,'loss':round(value,5),
                                    'seconds':round(time.monotonic()-before,3)})
            save();del batch,loss
            if args.max_microsteps and micro>=args.max_microsteps:break
        target=output/f'epoch-{epoch+1}'
        model.save_pretrained(target,safe_serialization=True)
        report['checkpoints'].append({'path':str(target),'epoch':epoch+1,'mean_train_loss':sum(losses)/len(losses)})
        save()
        if args.max_microsteps and micro>=args.max_microsteps:break
    report['status']='complete';save()
    print(json.dumps({k:v for k,v in report.items() if k!='steps'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--frames',type=int,default=8);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20261017);p.add_argument('--max-microsteps',type=int,default=0)
    a=p.parse_args()
    try:train(a)
    except Exception as e:
        # Detailed traces stay on the local host; never include annotation text in logs.
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        import traceback
        stack=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(e.__traceback__)]
        (out/'failure.json').write_text(json.dumps({'error_type':type(e).__name__,'message':str(e)[:300] if isinstance(e,ValueError) else 'Local training failed','stack':stack}))
        raise SystemExit(1)
