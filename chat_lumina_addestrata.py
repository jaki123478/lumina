"""Chat e voce direttamente da file locali, senza browser né server HTTP."""
import os
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from lumina_lora import attach,apply_state
ROOT=Path(__file__).resolve().parent


def load(path):
    folder=(ROOT/path).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError('Modello esterno al progetto')
    cfg=json.loads((folder/'adapter_config.json').read_text(encoding='utf-8'))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(cfg['base'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cfg['base'],local_files_only=True,dtype=torch.bfloat16 if device=='cuda' else torch.float32).to(device)
    attach(model,cfg['rank'],cfg['alpha']);apply_state(model,torch.load(folder/'adapter.pt',weights_only=True,map_location='cpu'));model.eval()
    return model,tok,cfg


def answer(model,tok,cfg,messages):
    turns=[{'role':'system','content':cfg['system']}]+messages[-12:]
    text=tok.apply_chat_template(turns,tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors='pt').to(next(model.parameters()).device)
    if inputs.input_ids.shape[1]>2048:raise ValueError('Conversazione troppo lunga: riavvia per una nuova sessione')
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=140,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--text');p.add_argument('--no-play',action='store_true');p.add_argument('--out',default='audio_neurale/lumina_addestrata.wav');args=p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True);torch.set_num_threads(4)
    path=args.model
    if not path:path=json.loads((ROOT/'config/lumina_trained_adapter.json').read_text(encoding='utf-8'))['path']
    print('Lumina locale addestrata — base Qwen 0.5B con adattatore; versione sperimentale.')
    model,tok,cfg=load(path);history=[]
    from lumina_voice_neural import synthesize,play
    while True:
        prompt=args.text or input('Tu > ').strip()
        if prompt.lower() in ('esci','exit','quit'):break
        if not prompt:continue
        history.append({'role':'user','content':prompt});reply=answer(model,tok,cfg,history)
        history.append({'role':'assistant','content':reply});print('Lumina:',reply,flush=True)
        output=synthesize(reply,ROOT/args.out)
        print('Audio:',output,flush=True)
        if not args.no_play:play(output)
        if args.text:break


if __name__=='__main__':main()
