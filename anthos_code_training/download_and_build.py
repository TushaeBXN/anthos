#!/usr/bin/env python3
import json, hashlib, re, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; NORM=ROOT/'normalized'; COMB=ROOT/'combined'; MAN=ROOT/'manifests'
SOURCES=[
('magicoder_oss_instruct_75k.jsonl','https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl','f988f14d3516a9f958eeea310f63903727c6d3463bc9cb435288b6ca4bb09cf9', 'MIT'),
('codefeedback_filtered_instruction.jsonl','https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/resolve/main/CodeFeedback-Filtered-Instruction.jsonl','6dd3f7797cd86a7e437de660ad259c50835812f8e309d893f09de77b2ee80063','Apache-2.0'),
('magicoder_evol_instruct_110k.jsonl','https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/resolve/main/data-evol_instruct-decontaminated.jsonl','99d8bab4c443050e5bb4bc339f709de8bcccf81cb83e15ef8d53369c5d8dc495','Apache-2.0'),
]

def fetch(name,url,sha):
    p=RAW/name
    if not p.exists():
        subprocess.run(['curl','-L','--fail','--retry','3','-o',str(p),url],check=True)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    if h.hexdigest()!=sha: raise SystemExit(f'Hash mismatch for {name}: {h.hexdigest()} != {sha}')
    return p

def norm_obj(obj, source, i):
    if source.startswith('magicoder_oss'):
        user=obj.get('problem',''); assistant=obj.get('solution','')
        msgs=[{'role':'user','content':user},{'role':'assistant','content':assistant}]
        meta={k:obj.get(k) for k in ('lang','index','raw_index','seed','openai_fingerprint') if k in obj}
    elif source.startswith('codefeedback'):
        user=obj.get('query',''); assistant=obj.get('answer','')
        msgs=[{'role':'user','content':user},{'role':'assistant','content':assistant}]
        meta={k:obj.get(k) for k in ('resource','lang') if k in obj}
    else:
        conv=obj.get('conversations') or obj.get('messages')
        if conv:
            msgs=[]
            for m in conv:
                role=m.get('role') or m.get('from')
                content=m.get('content') or m.get('value') or ''
                role={'human':'user','gpt':'assistant','system':'system'}.get(role,role)
                msgs.append({'role':role,'content':content})
        else:
            msgs=[{'role':'user','content':obj.get('instruction','')},{'role':'assistant','content':obj.get('output','')}]
        meta={k:v for k,v in obj.items() if k not in ('conversations','messages','instruction','output')}
    return {'id':f'{source}:{i:06d}','source':source,'messages':msgs,'metadata':meta}

def canon(s):
    s=re.sub(r'\s+',' ',s).strip().lower()
    return s

def build():
    for p in (NORM,COMB,MAN): p.mkdir(exist_ok=True)
    manifest={'sources':[],'counts':{},'notes':['Raw source files are downloaded from their official Hugging Face dataset repositories.','Combined output is exact-text deduplicated by normalized conversation text.']}
    allrows=[]
    for name,url,sha,license_ in SOURCES:
        p=fetch(name,url,sha)
        source=name[:-6]
        out=NORM/f'{source}.jsonl'; count=0
        with p.open('r',encoding='utf-8') as fi, out.open('w',encoding='utf-8') as fo:
            for i,line in enumerate(fi):
                if not line.strip(): continue
                obj=json.loads(line); row=norm_obj(obj,source,i)
                fo.write(json.dumps(row,ensure_ascii=False)+'\n'); allrows.append(row); count+=1
        manifest['counts'][source]=count
        manifest['sources'].append({'file':name,'url':url,'sha256':sha,'license':license_,'rows':count})
    seen=set(); kept=0; dup=0
    out=COMB/'anthos_code_sft_deduped.jsonl'
    with out.open('w',encoding='utf-8') as fo:
        for r in allrows:
            key=hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            if key in seen: dup+=1; continue
            seen.add(key); fo.write(json.dumps(r,ensure_ascii=False)+'\n'); kept+=1
    manifest['combined']={'raw_rows':len(allrows),'deduplicated_rows':kept,'duplicates_removed':dup,'output':str(out.relative_to(ROOT))}
    (MAN/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': build()
