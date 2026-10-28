"""Lezioni controllate, dati vari e selezione su generazioni; solo locale."""
import os
for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[k]='1'
os.environ['LUMINA_ALLOW_INTERNET']='0'
import argparse,ast,json,random,re,sys,time,shutil,operator,threading
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from lumina_next_runtime import ROOT,active_model_dir,prompt_ids,response_text,stop_sequences
from modeling_lumina_next import LuminaNextForCausalLM


def make_curriculum(seed=912):
    rng=random.Random(seed);rows=[]
    def add(q,a,cat,history=None):rows.append({'messages':(history or [])+[{'role':'user','content':q}],'assistant':a,'category':cat})
    for i in range(350):
        a,b=rng.randint(1,100),rng.randint(1,30);op=rng.choice(['+','-','*']);result={'+':a+b,'-':a-b,'*':a*b}[op]
        templates=['Calcola {}. Scrivi soltanto il risultato numerico.','Quanto fa {}? Rispondi solo con il numero.','Esegui {}. Solo il numero.','{} = ? Scrivi solo il risultato.']
        add(templates[i%4].format(f'{a} {op} {b}'),str(result),'calcolo')
    words=['lanterna','foglia','vela','argento','pietra','neve','ponte','tazza','stella','cobalto','platano','olmo','cedro','ambra','vento','porto','riva','fiore']
    for i in range(200):
        w=rng.choice(words);v=rng.choice(words);mode=i%5
        if mode==0:add(f'Ripeti esattamente questa parola, senza aggiunte: {w.upper()}',w.upper(),'istruzioni')
        if mode==1:add(f'Copia esattamente, senza virgolette: {w}.',w+'.','istruzioni')
        if mode==2:add(f'Scrivi la parola {w} due volte, separata da uno spazio. Non aggiungere altro.',w+' '+w,'istruzioni')
        if mode==3:add(f'Scrivi soltanto la prima parola di questa frase: {w.capitalize()} {v}.',w.capitalize(),'istruzioni')
        if mode==4:add(f'Restituisci esclusivamente questo JSON: {{"parola":"{w}"}}','{"parola":"'+w+'"}','istruzioni')
    names=['Arianna','Bruno','Carlo','Dina','Ettore','Flora','Giorgio','Irene','Livia','Nerio','Olga','Piero']
    colors=['giallo','viola','nero','bianco','rosa','verde','blu','rosso','arancione','turchese']
    for i in range(200):
        name=rng.choice(names);n=rng.randint(1,99);m=rng.randint(1,99);color=rng.choice(colors)
        if i%4==0:add(f'Il pacco contiene {n} penne e {m} gomme. Quante penne contiene? Solo il numero.',str(n),'estrazione')
        if i%4==1:add(f'La responsabile del laboratorio è {name}. Scrivi soltanto il nome della responsabile.',name,'estrazione')
        if i%4==2:add(f'Il tavolo è {color}, la sedia è grigia. Qual è il colore del tavolo? Una parola.',color,'estrazione')
        if i%4==3:
            cols=rng.sample(colors,3)
            add('Scrivi solo i colori '+', '.join(cols[:2])+' e '+cols[2]+', in questo ordine, separati da virgola e spazio.',', '.join(cols),'istruzioni')
    for i in range(200):
        name=rng.choice(names+words).capitalize();obj=rng.choice(['gatto','cane','coniglio'])
        add(f'Come si chiama il mio {obj}? Scrivi solo il nome.',name,'memoria',[
            {'role':'user','content':f'Il mio {obj} si chiama {name}.'},{'role':'assistant','content':'Va bene, ho letto.'}])
    facts=[('Francia','Parigi'),('Germania','Berlino'),('Giappone','Tokyo'),('Canada','Ottawa'),('Argentina','Buenos Aires'),
           ('Spagna','Madrid'),('Portogallo','Lisbona'),('Austria','Vienna'),('Polonia','Varsavia'),('Grecia','Atene'),
           ('Italia','Roma'),('Irlanda','Dublino'),('Belgio','Bruxelles'),('Norvegia','Oslo'),('Svezia','Stoccolma'),('Finlandia','Helsinki')]
    for country,city in facts:
        for q in [f'Qual è la capitale di {country}?',f'Indicami la capitale di {country}.',f'Capitale di {country}?',f'Qual è la capitale di {country}? Rispondi solo con il nome.']:
            add(q,city,'conoscenze')
    for q,a in [
        ('Come si scrive la formula chimica dell’acqua?','H2O'),('Quanti centimetri formano un metro?','100'),
        ('Come si scrive il plurale della parola città?','città'),('Qual è il più grande oceano della Terra?','Oceano Pacifico'),
        ('Come ti chiami?','Sono Lumina, un assistente locale in lingua italiana.'),
        ('Chi sei?','Sono Lumina, un assistente locale in lingua italiana.'),
        ('Ciao, presentati.','Sono Lumina, un assistente locale in lingua italiana.'),
        ('Perché le piante fanno la fotosintesi?','Le piante usano la luce per trasformare acqua e anidride carbonica in zuccheri, liberando ossigeno.'),
        ('Come correggo "io ho andato al mare"?','Io sono andato al mare.'),
        ('Come si corregge "noi abbiamo andati a scuola"?','Noi siamo andati a scuola.'),
        ('Quale lago italiano ha la superficie maggiore?','Il lago di Garda.'),
        ('Puoi prevedere chi vincerà una gara futura?','Non posso prevedere con certezza il risultato di un evento futuro.'),
        ('Qual è il nome di una persona che non ti ho mai descritto?','Non lo so: non mi hai fornito questa informazione.'),
        ('Puoi leggere il numero nel mio quaderno chiuso?','Non posso vederlo: non ho accesso al tuo quaderno.'),
        ('Il testo dice che un pacco è arrivato. Quanto pesa?','Il peso non è specificato nel testo.')]:add(q,a,'conoscenze')
    days=['lunedì','martedì','mercoledì','giovedì','venerdì','sabato','domenica']
    for i in range(120):
        start=rng.randrange(7);delta=rng.randint(1,6)
        add(f'Oggi è {days[start]}. Che giorno sarà fra {delta} giorni? Rispondi solo con il giorno.',days[(start+delta)%7],'logica')
    for name,body,args in [('doppio','2 * n','n'),('triplo','3 * n','n'),('cubo','x ** 3','x'),('quadrato','x ** 2','x'),('aggiungi','a + b','a, b'),('somma','a + b','a, b'),('maggiore','a if a > b else b','a, b')]:
        for q in [f'Scrivi soltanto il codice Python della funzione {name}({args}) che restituisce {body}.',f'Definisci {name}({args}) in Python: risultato {body}. Solo codice.']:
            add(q,f'```python\ndef {name}({args}):\n    return {body}\n```','codice')
    # Conserva l'intero vecchio collaudo come controllo, mai come lezione esatta.
    reserved=json.loads((ROOT/'evaluation/general_cases.json').read_text(encoding='utf-8'))['tests']
    forbidden={json.dumps(t['history']+[{'role':'user','content':t['prompt']}],sort_keys=True,ensure_ascii=False) for t in reserved}
    unique={json.dumps(r['messages'],sort_keys=True,ensure_ascii=False):r for r in rows}
    return [r for key,r in unique.items() if key not in forbidden]


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=800);p.add_argument('--output',default='outputs/verified_general_'+time.strftime('%Y%m%d_%H%M%S'));p.add_argument('--teacher-lessons',type=int,default=24);args=p.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output deve essere locale nel progetto')
    out.mkdir(parents=True,exist_ok=False)
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    import msvcrt
    lock=(ROOT/'outputs/qwen_repair.lock').open('a+b');lock.seek(0,2)
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    live_log=(ROOT/'live_training.log').open('a',encoding='utf-8')
    def log(msg):print(msg,flush=True);live_log.write('\n[QWEN VERIFICATO] '+msg+'\n');live_log.flush()
    rows=make_curriculum();(out/'train.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    # Qwen propone lezioni chiuse: il riferimento esatto decide l'accettazione.
    import requests
    session=requests.Session();session.trust_env=False;teacher=[]
    candidates=[r for r in rows if r['category'] in ('calcolo','istruzioni','estrazione','memoria')]
    random.Random(371).shuffle(candidates)
    for i,row in enumerate(candidates[:args.teacher_lessons],1):
        d=session.post('http://127.0.0.1:8000/v1/chat/completions',json={'model':'lumina-assisted','messages':row['messages'],'max_tokens':120,'temperature':0,'purpose':'teacher_lesson'},timeout=240).json()
        answer=d.get('choices',[{}])[0].get('message',{}).get('content','').strip()
        accepted=answer==row['assistant'] and d.get('backend',{}).get('finished',True)
        teacher.append({'prompt':row['messages'],'expected':row['assistant'],'answer':answer,'accepted':accepted,'backend':d.get('backend')})
        log(f'LEZIONE {i}/{args.teacher_lessons} '+('VERIFICATA' if accepted else 'SCARTATA')+' | '+row['messages'][-1]['content']+' | risposta: '+answer)
    (out/'teacher.json').write_text(json.dumps(teacher,ensure_ascii=False,indent=2),encoding='utf-8')
    from qwen_repair_trainer import training_sequence
    from qwen_precision import precision_instruction
    tok=ByteLevelBPETokenizer(str(ROOT/'tokenizer_lumina_next/vocab.json'),str(ROOT/'tokenizer_lumina_next/merges.txt'))
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4);torch.manual_seed(371);random.seed(371)
    source=active_model_dir();model=LuminaNextForCausalLM.from_pretrained(source).to(device)
    amp=lambda:torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda')
    tests=json.loads((ROOT/'evaluation/general_cases.json').read_text(encoding='utf-8'))['tests']
    from evaluation.code_checks import verify_code
    def evaluate():
        model.eval();results=[]
        with torch.inference_mode(),amp():
            for c in tests:
                ids=prompt_ids(tok,c['history']+[{'role':'user','content':c['prompt']}]);gen=model.generate(torch.tensor([ids],device=device),temperature=0,max_new_tokens=100,stop_token_sequences=stop_sequences(tok));answer=response_text(tok,gen[0,len(ids):].tolist())
                if c['kind']=='exact':ok=answer==c['expected']
                elif c['kind']=='code':ok=verify_code(answer,c['expected'],c['extra'])[0]
                elif c['area']=='Limiti':
                    lower=answer.lower();ok=any(s in lower for s in ['non lo so','non so','non conosco','non posso','non è possibile','non è specificat','non è indicat'])
                    if '2055' not in c['prompt'] and 'futuro' in lower:ok=False
                    if any(s in c['prompt'] for s in ['quaderno','peso']) and 'nome' in lower:ok=False
                else:ok=any(v in answer.lower() for v in c['expected'])
                results.append({'area':c['area'],'prompt':c['prompt'],'response':answer,'pass':ok})
        return {'pass':sum(x['pass'] for x in results),'total':len(results),'responses':results}
    baseline=evaluate();log(f'BASELINE NATIVA: {baseline["pass"]}/60')
    report={'source':str(source),'train_samples':len(rows),'teacher_accepted':sum(x['accepted'] for x in teacher),'teacher_total':len(teacher),'baseline':baseline,'history':[],'activated':False}
    save=lambda:(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();pools={}
    for row in rows:
        seq=training_sequence(tok,row)
        if seq is not None:pools.setdefault(row['category'],[]).append(seq)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);best=baseline['pass'];started=time.time()
    categories=sorted(pools)
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=[random.choice(pools[categories[(step+i)%len(categories)]]) for i in range(4)];length=max(len(x) for x,y in batch)
        xx=torch.full((4,length),2,device=device,dtype=torch.long);yy=torch.full((4,length),-100,device=device,dtype=torch.long)
        for i,(x,y) in enumerate(batch):xx[i,:len(x)]=torch.tensor(x,device=device);yy[i,:len(y)]=torch.tensor(y,device=device)
        with amp():loss=F.cross_entropy(model(xx).float().reshape(-1,32000),yy.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):raise RuntimeError('Loss non finita')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%50==0:log(f'TRAIN {step}/{args.steps} | loss {float(loss.detach()):.4f}')
        if step%200==0 or step==args.steps:
            metrics=evaluate();report['history'].append({'step':step,**metrics});log(f'CONTROLLO NATIVO {step}: {metrics["pass"]}/60')
            # Mai confondere questa selezione con un test finale indipendente.
            if metrics['pass']>=baseline['pass']+5 and metrics['pass']>best:
                best=metrics['pass'];model.save_pretrained(out/'candidate')
                for file in ('vocab.json','merges.txt'):shutil.copy2(ROOT/'tokenizer_lumina_next'/file,out/'candidate'/file)
                report['candidate_step']=step;report['candidate_score']=best
            save()
    report['elapsed_s']=time.time()-started
    # Nessuna attivazione automatica: il test finale con domande nuove precede la selezione.
    save();log('CICLO COMPLETATO. Candidato separato; attivazione solo dopo prova finale. Report: '+str(out/'report.json'))

if __name__=='__main__':main()
