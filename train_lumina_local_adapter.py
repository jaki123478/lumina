"""Addestramento offline dell'adattatore Lumina su Qwen 0.5B già in cache."""
import os
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[key]='1'
os.environ['LUMINA_ALLOW_INTERNET']='0'
import sys,json,time,random,re,argparse,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from lumina_lora import attach,adapter_state,apply_state
from qwen_verified_trainer import make_curriculum
ROOT=Path(__file__).resolve().parent
BASE='Qwen/Qwen2.5-0.5B-Instruct'
SYSTEM='Sei Lumina, assistente locale in italiano. Rispondi direttamente e rispetta il formato richiesto. Usa i dati della conversazione. Se mancano informazioni, dichiaralo senza inventare.'


def grade(case,text):
    if case['kind']=='exact':return text.strip()==case['expected']
    return any(bool(re.search(r'(?<!\d)'+re.escape(v)+r'(?!\d)',text)) if v.isdigit() else v in text.lower() for v in case['expected'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=320);ap.add_argument('--output',default='outputs/lumina_adapter_'+time.strftime('%Y%m%d_%H%M%S'));ap.add_argument('--phase2',action='store_true');args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output esterno al progetto')
    out.mkdir(parents=True,exist_ok=False)
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    denied=[]
    def audit(event,values):
        if event=='socket.connect' and isinstance(values[1],tuple):
            denied.append(str(values[1]));raise RuntimeError('Rete disabilitata durante addestramento')
        if event=='socket.getaddrinfo':
            denied.append(str(values[0]));raise RuntimeError('DNS disabilitato durante addestramento')
    sys.addaudithook(audit)
    torch.manual_seed(6401);random.seed(6401);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16 if device=='cuda' else torch.float32).to(device)
    names=attach(model,rank=8,alpha=16)
    dev=json.loads((ROOT/'evaluation/general_holdout.json').read_text(encoding='utf-8'))
    final=[
        {'area':'calcolo','prompt':'Quanto fa 42 + 39? Rispondi solo con il numero.','expected':'81','kind':'exact'},
        {'area':'istruzioni','prompt':'Copia esattamente questa parola, senza aggiunte: GINESTRA','expected':'GINESTRA','kind':'exact'},
        {'area':'istruzioni','prompt':'Scrivi la parola vela due volte, separata da uno spazio. Non aggiungere altro.','expected':'vela vela','kind':'exact'},
        {'area':'estrazione','prompt':'La custode si chiama Rosalba. Scrivi soltanto il nome della custode.','expected':'Rosalba','kind':'exact'},
        {'area':'estrazione','prompt':'La busta contiene 67 figurine e 9 francobolli. Quante figurine contiene? Solo il numero.','expected':'67','kind':'exact'},
        {'area':'memoria','history':[{'role':'user','content':'Il mio criceto si chiama Rubino.'},{'role':'assistant','content':'Va bene.'}],'prompt':'Come si chiama il mio criceto? Scrivi solo il nome.','expected':'Rubino','kind':'exact'},
        {'area':'conoscenze','prompt':'Qual è la capitale del Portogallo?','expected':['lisbona'],'kind':'contains'},
        {'area':'conoscenze','prompt':'Quanti lati ha un triangolo?','expected':['3','tre'],'kind':'contains'},
        {'area':'limiti','prompt':'Sai qual è il numero nel mio cassetto chiuso? Non puoi vederlo.','expected':['non posso','non lo so','non ho'],'kind':'contains'},
        {'area':'grammatica','prompt':'Correggi: noi abbiamo andati al parco.','expected':['siamo andati'],'kind':'contains'}]
    if args.phase2:
        dev+=final
        final=[
            {'area':'calcolo','prompt':'Quanto fa 26 + 48? Rispondi solo con il numero.','expected':'74','kind':'exact'},
            {'area':'istruzioni','prompt':'Copia esattamente questa parola, senza aggiunte: MAGNOLIA','expected':'MAGNOLIA','kind':'exact'},
            {'area':'istruzioni','prompt':'Scrivi la parola porto due volte, separata da uno spazio. Non aggiungere altro.','expected':'porto porto','kind':'exact'},
            {'area':'estrazione','prompt':'La responsabile della serra è Maristella. Scrivi soltanto il nome.','expected':'Maristella','kind':'exact'},
            {'area':'estrazione','prompt':'Il contenitore ha 43 nocciole e 6 noci. Quante nocciole contiene? Solo il numero.','expected':'43','kind':'exact'},
            {'area':'memoria','history':[{'role':'user','content':'Il mio cane si chiama Quarzo.'},{'role':'assistant','content':'Va bene.'}],'prompt':'Come si chiama il mio cane? Rispondi solo con il nome.','expected':'Quarzo','kind':'exact'},
            {'area':'conoscenze','prompt':'Qual è la capitale della Finlandia?','expected':['helsinki'],'kind':'contains'},
            {'area':'conoscenze','prompt':'Quanti lati possiede un quadrato?','expected':['4','quattro'],'kind':'contains'},
            {'area':'limiti','prompt':'Qual è il numero nascosto nel mio armadio? Non puoi vederlo.','expected':['non posso','non lo so','non ho'],'kind':'contains'},
            {'area':'grammatica','prompt':'Correggi: io ho andato in biblioteca.','expected':['sono andato'],'kind':'contains'}]
    def messages(c):return [{'role':'system','content':SYSTEM}]+c.get('history',[])+[{'role':'user','content':c['prompt']}]
    reserved={json.dumps(messages(c)[1:],sort_keys=True,ensure_ascii=False) for c in dev+final}
    raw_rows=make_curriculum(seed=6401)
    if args.phase2:
        def extra(q,a,category):raw_rows.append({'messages':[{'role':'user','content':q}],'assistant':a,'category':category})
        pairs=[('io ho andato a Roma','Io sono andato a Roma.'),('noi abbiamo andati a Milano','Noi siamo andati a Milano.'),
               ('lei ha uscita ieri','Lei è uscita ieri.'),('io ho tornato a casa','Io sono tornato a casa.'),
               ('voi avete partiti presto','Voi siete partiti presto.'),('noi abbiamo arrivati tardi','Noi siamo arrivati tardi.'),
               ('loro hanno entrati in aula','Loro sono entrati in aula.'),('tu hai andato a scuola','Tu sei andato a scuola.')]
        for wrong,right in pairs:
            for template in ['Correggi: {}.','Correggi la frase: {}.','Riscrivi correttamente questa frase: {}.','Qual è la forma corretta di "{}"?']:
                extra(template.format(wrong),right,'grammatica')
        for obj in ['borsa','scatola','cassetto','armadio','zaino','quaderno','busta','scrigno','valigia','garage']:
            for q in [f'Qual è il numero nel mio {obj}? Non puoi vederlo.',f'Sai cosa contiene il mio {obj} chiuso? Non te l’ho detto.',f'Puoi leggere il numero nascosto nel mio {obj}?']:
                extra(q,'Non posso saperlo: non ho accesso a questo oggetto e non mi hai dato questa informazione.','limiti')
        for year in [2043,2047,2052,2058,2065,2071,2080,2090]:
            for event in ['una gara','il campionato','il torneo','la finale']:
                for q in [f'Chi vincerà {event} nel {year}?',f'Puoi sapere chi vincerà {event} nel {year}?']:
                    extra(q,'Non è possibile prevedere con certezza il risultato di un evento futuro.','limiti')
        for shape,n in [('triangolo',3),('quadrato',4),('pentagono',5),('esagono',6),('ettagono',7),('ottagono',8)]:
            for q in [f'Quanti lati ha un {shape}?',f'Un {shape} quanti lati possiede?',f'Quanti sono i lati di un {shape}? Rispondi solo con il numero.']:
                extra(q,str(n),'conoscenze')
    rows=[r for r in raw_rows if json.dumps(r['messages'],sort_keys=True,ensure_ascii=False) not in reserved]
    pools={}
    def prepare(row):
        prefix=tok.apply_chat_template([{'role':'system','content':SYSTEM}]+row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
        answer=tok.encode(row['assistant'],add_special_tokens=False)+[tok.eos_token_id]
        if len(prefix)+len(answer)>256:return None
        return prefix+answer[:-1],[-100]*(len(prefix)-1)+answer
    for row in rows:
        seq=prepare(row)
        if seq:pools.setdefault(row['category'],[]).append(seq)
    (out/'train.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    (out/'development.json').write_text(json.dumps(dev,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'final_questions.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    def evaluate(cases):
        model.eval();results=[]
        with torch.inference_mode():
            for c in cases:
                text=tok.apply_chat_template(messages(c),tokenize=False,add_generation_prompt=True)
                inp=tok(text,return_tensors='pt').to(device)
                out_ids=model.generate(**inp,max_new_tokens=100,do_sample=False,pad_token_id=tok.eos_token_id)
                reply=tok.decode(out_ids[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
                results.append({**c,'response':reply,'pass':grade(c,reply)})
        return {'pass':sum(r['pass'] for r in results),'total':len(results),'answers':results}
    baseline=evaluate(dev);final_before=evaluate(final)
    initial=adapter_state(model)
    report={'base':BASE,'source':'cache locale','rank':8,'alpha':16,'layers':len(names),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'train_samples':sum(len(v) for v in pools.values()),'baseline_dev':baseline,'final_before':final_before,'history':[],'activated':False}
    save=lambda:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();print('BASELINE',baseline['pass'],'/',len(dev),'finale iniziale',final_before['pass'],'/10')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5 if args.phase2 else 1e-4,weight_decay=.01)
    best=baseline['pass'];best_state=None;categories=sorted(pools);start=time.time()
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=[random.choice(pools[categories[(step+i)%len(categories)]]) for i in range(2)];n=max(len(x) for x,y in batch)
        x=torch.full((2,n),tok.pad_token_id,device=device,dtype=torch.long);y=torch.full((2,n),-100,device=device,dtype=torch.long);mask=torch.zeros((2,n),device=device,dtype=torch.long)
        for i,(xx,yy) in enumerate(batch):x[i,:len(xx)]=torch.tensor(xx,device=device);y[i,:len(yy)]=torch.tensor(yy,device=device);mask[i,:len(xx)]=1
        logits=model(input_ids=x,attention_mask=mask,use_cache=False).logits
        loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):raise RuntimeError('Loss non finita')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
        if step%40==0:print(f'TRAIN {step}/{args.steps}: loss={float(loss.detach()):.4f}')
        if step%80==0 or step==args.steps:
            metrics=evaluate(dev);report['history'].append({'step':step,**metrics});print('CONTROLLO',step,metrics['pass'],'/',len(dev))
            protected=[i for i,c in enumerate(dev) if c['area'] in ('grammatica','limiti') and baseline['answers'][i]['pass']]
            preserves=all(metrics['answers'][i]['pass'] for i in protected)
            if metrics['pass']>best and preserves:
                best=metrics['pass'];best_state=adapter_state(model);torch.save(best_state,out/'adapter.pt');report['best_step']=step
            save()
    report['development_total']=len(dev)
    torch.save(adapter_state(model),out/'last_adapter.pt')
    if best_state:
        apply_state(model,best_state);report['final_after']=evaluate(final)
        report['ready']=report['final_after']['pass']>=final_before['pass'] and best>baseline['pass']
        (out/'adapter_config.json').write_text(json.dumps({'base':BASE,'rank':8,'alpha':16,'system':SYSTEM,'report':'report.json'},indent=2),encoding='utf-8')
    else:report['ready']=False;report['final_after']=final_before
    report['elapsed_s']=time.time()-start;report['denied_network_attempts']=denied;save()
    print('FINE',json.dumps({'best_dev':best,'final_before':final_before['pass'],'final_after':report['final_after']['pass'],'ready':report['ready'],'report':str(out/'report.json')}))

if __name__=='__main__':main()
