"""Distillazione locale, bilanciata e verificata. Mai sovrascrivere i pesi sorgente."""
import os
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key] = '1'
os.environ['LUMINA_ALLOW_INTERNET'] = '0'
import argparse, ast, gc, hashlib, json, random, re, shutil, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from modeling_lumina_next import LuminaNextForCausalLM
from lumina_next_runtime import prompt_ids, stop_sequences, response_text, ROOT
from qwen_precision import precision_instruction

RESERVED_QUESTIONS = {
    'Ciao, chi sei?', 'Quanto fa 17 + 28? Rispondi solo con il numero.',
    'Quanto fa 23 per 7? Rispondi solo con il numero.', "Qual è il lago più grande d'Italia?",
    'Correggi la frase: io ho andato al mare.', 'Spiega brevemente la fotosintesi.',
    'Se oggi è martedì, che giorno sarà tra tre giorni?',
    'Scrivi soltanto questi tre colori separati da virgole: rosso, verde, blu.',
    'Nel testo: "Marta ha 4 mele e Luca ha 9 pere". Quante pere ha Luca?',
    'Chi vincerà lo scudetto nel 2040?', 'Come si chiama il mio gatto?',
    'Scrivi una funzione Python quadrato(x) che restituisce il quadrato di x.'}


def validate_teacher(answer, finished=True):
    if not finished or not answer.strip() or any(s in answer for s in ('<|im_start|>', '<|im_end|>')):
        return False
    if answer.count('```') % 2:
        return False
    for code in re.findall(r'```(?:python)?\s*\n(.*?)```', answer, flags=re.S):
        try:
            ast.parse(code)
        except SyntaxError:
            return False
    return True


def example(question, answer, category, history=None):
    return {'messages': (history or []) + [{'role':'user','content':question}],
            'assistant':answer, 'category':category}


def curriculum(seed):
    rng = random.Random(seed)
    train, valid = [], []
    # Separazione per coppia numerica, mai per duplicati della stessa domanda.
    pairs = [(a,b) for a in range(2,60) for b in range(2,40)]
    rng.shuffle(pairs)
    for i,(a,b) in enumerate(pairs[:220]):
        dest = valid if i < 20 else train
        op = '+' if i % 2 or b>14 else '*'
        answer = str(a+b if op=='+' else a*b)
        dest.append(example(f'Calcola {a} {op} {b}. Rispondi solo con il numero.',answer,'calcolo'))
    colors = ['rosso','verde','blu','giallo','viola','bianco','nero','arancione','rosa']
    for i in range(84):
        words = rng.sample(colors,3)
        dest = valid if i < 8 else train
        templates=['Ripeti soltanto: {}.', 'Scrivi soltanto questi tre colori separati da virgole: {}.',
                   'Elenca solo i colori seguenti, separati da virgole: {}.', 'Rispondi esattamente con: {}.']
        dest.append(example(templates[i%len(templates)].format(', '.join(words)), ', '.join(words),'istruzioni'))
    for i in range(84):
        name = rng.choice(['Ada','Luca','Marta','Nora','Paolo','Emma'])
        n = rng.randint(2,80)
        obj = rng.choice(['pere','mele','scatole','matite'])
        dest = valid if i < 8 else train
        dest.append(example(f'Nel testo: "{name} ha {n} {obj}". Quante {obj} ha {name}? Rispondi solo con il numero.',str(n),'estrazione'))
    for i in range(64):
        name = rng.choice(['Zeno','Milo','Luna','Nube','Tito','Argo','Sole','Briciola'])
        animal = rng.choice(['gatto','cane','coniglio'])
        dest = valid if i < 8 else train
        dest.append(example(f'Come si chiama il mio {animal}? Rispondi solo con il nome.',name,'memoria',[
            {'role':'user','content':f'Il mio {animal} si chiama {name}.'},
            {'role':'assistant','content':f'Il tuo {animal} si chiama {name}.'}]))
    basics = [
        ('Ciao!','Ciao! Sono Lumina. Come posso aiutarti?'),
        ('Come ti chiami?','Sono Lumina, un assistente locale in lingua italiana.'),
        ('Chi sei?','Sono Lumina, un assistente locale in lingua italiana.'),
        ('Come si chiama il mio gatto?','Non lo so: non mi hai ancora detto il suo nome.'),
        ('Come si chiama il mio cane?','Non lo so: non mi hai ancora detto il suo nome.'),
        ('Qual è il mio numero preferito?','Non lo so: non mi hai ancora detto il tuo numero preferito.'),
        ('Grazie per il tuo aiuto.','Prego! Sono qui per aiutarti.'),
        ('Puoi funzionare senza Internet?','Sì, uso modelli e dati presenti sul computer.'),
    ]
    for q,a in basics:
        train.append(example(q,a,'identita'))
    for q in ['Ciao, puoi presentarti?', 'Dimmi il tuo nome.', 'Come ti chiami e cosa fai?', 'Presentati brevemente.', 'Qual è il tuo nome?', 'Tu chi sei?', 'Sei Lumina?', 'Ciao Lumina, chi sei?', 'Mi dici chi sei?', 'Parlami di te.']:
        train.append(example(q,'Sono Lumina, un assistente locale in lingua italiana.','identita'))
    for animal in ['cane','coniglio','pappagallo','criceto']:
        for form in ['Come si chiama il mio {}?', 'Sai il nome del mio {}?', 'Qual è il nome del mio {}?']:
            train.append(example(form.format(animal),'Non lo so: non mi hai ancora detto il suo nome.','identita'))
    # Replay di contenuti corretti, senza sovracampionare il benchmark.
    for line in (ROOT/'data/distilled_dialogues.jsonl').open(encoding='utf-8'):
        try:
            r=json.loads(line)
            q,a=r.get('user',''),r.get('assistant','')
            if len(a)<450 and len(q)<350 and validate_teacher(a) and '```' not in a:
                if rng.random()<0.004:
                    train.append(example(q,a,'replay'))
        except (ValueError,TypeError):
            continue
    # Impedisce che i campioni casuali identici contaminino la validazione.
    fingerprint = lambda x: json.dumps(x['messages'],sort_keys=True,ensure_ascii=False)
    held = {fingerprint(x) for x in valid}
    unique = {}
    for row in train:
        if fingerprint(row) not in held and row['messages'][-1]['content'] not in RESERVED_QUESTIONS:
            unique[fingerprint(row)]=row
    return list(unique.values()),valid


def training_sequence(tok, row, limit=384):
    prefix=prompt_ids(tok,row['messages'],budget=limit)
    answer=tok.encode(row['assistant']+'<|im_end|>').ids+[1]
    ids=prefix+answer
    if len(ids)>limit:
        return None  # Mai insegnare risposte troncate.
    labels=[-100]*(len(prefix)-1)+answer
    return ids[:-1],labels


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=240)
    ap.add_argument('--teacher-lessons',type=int,default=12)
    ap.add_argument('--teacher-cache',default='')
    ap.add_argument('--patience',type=int,default=3,help='Interrompi dopo N verifiche senza un candidato migliore')
    ap.add_argument('--output',default='outputs/qwen_repair_'+time.strftime('%Y%m%d_%H%M%S'))
    args=ap.parse_args()
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT): raise ValueError('Output esterno al progetto')
    out.mkdir(parents=True,exist_ok=True)
    # Il blocco del sistema operativo viene rilasciato anche se il processo termina.
    lock=(ROOT/'outputs/qwen_repair.lock').open('a+b')
    if lock.tell()==0: lock.write(b'0');lock.flush()
    lock.seek(0)
    import msvcrt
    try:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close()
        raise RuntimeError('Un altro ciclo di riparazione Qwen è già in corso')
    if (out/'report.json').exists(): raise FileExistsError('Usare un nuovo output per preservare il ciclo precedente')
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    torch.manual_seed(73); random.seed(73); torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    amp=lambda: torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda')
    tok=ByteLevelBPETokenizer(str(ROOT/'tokenizer_lumina_next/vocab.json'),str(ROOT/'tokenizer_lumina_next/merges.txt'))
    train,valid=curriculum(73)
    teacher_report=[]
    if args.teacher_cache:
        teacher_report=json.loads((ROOT/args.teacher_cache).read_text(encoding='utf-8'))['teacher']
        for row in teacher_report:
            if row['accepted'] and validate_teacher(row['answer']):
                train.append(example(row['question'],row['answer'],'qwen_validato'))
    if args.teacher_lessons:
        from transformers import AutoTokenizer,AutoModelForCausalLM
        teacher_id='Qwen/Qwen2.5-Coder-1.5B-Instruct'
        tt=AutoTokenizer.from_pretrained(teacher_id,local_files_only=True)
        teacher=AutoModelForCausalLM.from_pretrained(teacher_id,local_files_only=True,dtype=torch.bfloat16 if device=='cuda' else torch.float32).to(device).eval()
        questions=[
            'Scrivi solo il codice Python della funzione somma(a, b).',
            'Scrivi solo il codice Python della funzione quadrato(x).',
            'Scrivi solo il codice Python della funzione filtra_pari(numeri).',
            'Scrivi solo il codice Python della funzione massimo(a, b).',
            'Spiega la differenza fra lista e tupla in Python in due frasi.',
            'Spiega cosa significa KeyError in Python in due frasi.',
            'Spiega in due frasi a cosa serve return in Python.',
            'Spiega la fotosintesi in due frasi semplici.',
            'Spiega in due frasi perché la Luna è luminosa.',
            'Spiega la differenza tra una pila e una coda in due frasi.',
            'Correggi la frase: noi abbiamo andati a scuola.',
            'Spiega in due frasi perché non puoi conoscere un risultato sportivo futuro.'
        ]
        for index,q in enumerate(questions[:args.teacher_lessons]):
            chat=tt.apply_chat_template([{'role':'system','content':precision_instruction()+'\nRispondi in italiano, in modo breve e completo. Niente preamboli. Per codice usa un blocco python completo; altrimenti al massimo 45 parole.'},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
            inp=tt(chat,return_tensors='pt').to(device)
            with torch.inference_mode():
                gen=teacher.generate(**inp,max_new_tokens=240,do_sample=False,pad_token_id=tt.eos_token_id)
            ids=gen[0,inp.input_ids.shape[1]:]
            answer=tt.decode(ids,skip_special_tokens=True).strip()
            accepted=validate_teacher(answer,len(ids)<240 or int(ids[-1])==tt.eos_token_id)
            if 'solo il codice' in q and not re.search(r'```python\s*\n.*?\bdef\b',answer,re.S): accepted=False
            teacher_report.append({'question':q,'answer':answer,'accepted':accepted})
            if accepted: train.append(example(q,answer,'qwen_validato'))
            print(f'QWEN {index+1}/{min(args.teacher_lessons,len(questions))}: '+('ACCETTATA' if accepted else 'SCARTATA'))
        del teacher,tt,inp,gen; gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
    # Confronto prima/dopo su domande non incluse nelle lezioni.
    probes=[
        ('Ciao, chi sei?', ['lumina']),
        ('Quanto fa 17 + 28? Rispondi solo con il numero.', ['45']),
        ('Quanto fa 23 per 7? Rispondi solo con il numero.', ['161']),
        ("Qual è il lago più grande d'Italia?", ['garda']),
        ('Correggi la frase: io ho andato al mare.', ['sono andato']),
        ('Spiega brevemente la fotosintesi.', ['piante','luce']),
        ('Se oggi è martedì, che giorno sarà tra tre giorni?', ['venerdì']),
        ('Scrivi soltanto questi tre colori separati da virgole: rosso, verde, blu.', ['rosso, verde, blu']),
        ('Nel testo: "Marta ha 4 mele e Luca ha 9 pere". Quante pere ha Luca?', ['9']),
        ('Chi vincerà lo scudetto nel 2040?', ['non']),
        ('Come si chiama il mio gatto?', ['non']),
        ('Scrivi una funzione Python quadrato(x) che restituisce il quadrato di x.', ['def','return']),
    ]
    valid_seq=[training_sequence(tok,row) for row in valid]
    def evaluate(model, generate=False):
        model.eval(); losses=[]; replies=[]
        with torch.inference_mode(),amp():
            for seq in valid_seq:
                if seq is None: continue
                x,y=seq
                logits=model(torch.tensor([x],device=device))
                loss=F.cross_entropy(logits.float().reshape(-1,32000),torch.tensor(y,device=device),ignore_index=-100)
                losses.append(float(loss))
            if generate:
                for q,keys in probes:
                    ids=prompt_ids(tok,[{'role':'user','content':q}])
                    gen=model.generate(torch.tensor([ids],device=device),max_new_tokens=90,temperature=0,stop_token_sequences=stop_sequences(tok))
                    text=response_text(tok,gen[0,len(ids):].tolist())
                    replies.append({'prompt':q,'response':text,'pass':all(k in text.lower() for k in keys)})
        return {'validation_loss':sum(losses)/len(losses),'probes':replies,'probe_pass':sum(r['pass'] for r in replies)}
    from lumina_next_runtime import active_model_dir
    sources=[active_model_dir()]
    active_config=ROOT/'config/lumina_active_model.json'
    previous_selection=active_config.read_bytes() if active_config.exists() else None
    latest=out/'base_checkpoint.pt'
    baselines=[]
    model=LuminaNextForCausalLM.from_pretrained(sources[0]).to(device)
    baseline=evaluate(model,True); baselines.append({'source':str(sources[0]),**baseline})
    print('BASELINE EXPORT',json.dumps(baseline,ensure_ascii=False))
    if latest.exists():
        model.load_state_dict(torch.load(latest,map_location='cpu',weights_only=True))
        recent=evaluate(model,True); baselines.append({'source':str(latest),**recent})
        print('BASELINE ULTIMO QWEN',json.dumps(recent,ensure_ascii=False))
        if recent['validation_loss'] < baseline['validation_loss'] and recent['probe_pass'] >= baseline['probe_pass']:
            baseline=recent
        else:
            del model; gc.collect()
            model=LuminaNextForCausalLM.from_pretrained(sources[0]).to(device)
    for file in ('vocab.json','merges.txt'):
        shutil.copy2(ROOT/'tokenizer_lumina_next'/file,out/file)
    prepared=[seq for row in train if (seq:=training_sequence(tok,row)) is not None]
    (out/'train.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in train),encoding='utf-8')
    (out/'validation.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in valid),encoding='utf-8')
    report={'seed':73,'offline':True,'teacher':teacher_report,'baselines':baselines,'train_samples':len(prepared),'validation_samples':len(valid_seq),'history':[],'baseline':baseline,'activated':False}
    def save_report(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save_report()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    best_loss=baseline['validation_loss']; best=None; started=time.time(); stale_checks=0
    for step in range(1,args.steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        # Alterna categorie per evitare che l'aritmetica cancelli identità e codice.
        categories=sorted({r['category'] for r in train if training_sequence(tok,r) is not None})
        category=categories[(step-1)%len(categories)]
        pool=[training_sequence(tok,r) for r in train if r['category']==category]
        pool=[s for s in pool if s is not None]
        batch=random.choices(pool,k=2); n=max(len(x) for x,y in batch)
        xx=torch.full((2,n),2,device=device,dtype=torch.long)
        yy=torch.full((2,n),-100,device=device,dtype=torch.long)
        for i,(x,y) in enumerate(batch): xx[i,:len(x)]=torch.tensor(x,device=device); yy[i,:len(y)]=torch.tensor(y,device=device)
        with amp():
            logits=model(xx)
            loss=F.cross_entropy(logits.float().reshape(-1,32000),yy.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss): raise RuntimeError('Loss non finita: checkpoint originale preservato')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if step%20==0: print(f'TRAIN {step}/{args.steps} loss={float(loss.detach()):.4f} elapsed={time.time()-started:.1f}s')
        if step%60==0 or step==args.steps:
            metrics=evaluate(model,True)
            report['history'].append({'step':step,**metrics}); save_report()
            print('VALIDAZIONE',step,json.dumps(metrics,ensure_ascii=False))
            # Promozione richiede migliore loss su esempi separati E più risposte corrette.
            preserves_basics = all(not old['pass'] or new['pass'] for old,new in zip(baseline['probes'],metrics['probes']))
            if metrics['validation_loss']<best_loss and metrics['probe_pass']>baseline['probe_pass'] and preserves_basics:
                best_loss=metrics['validation_loss']; best=metrics
                model.save_pretrained(out/'candidate')
                for file in ('vocab.json','merges.txt'): shutil.copy2(out/file,out/'candidate'/file)
                stale_checks=0
            else:
                stale_checks+=1
            if stale_checks>=args.patience:
                print('Arresto anticipato: nessun miglioramento verificato nelle ultime valutazioni.')
                break
    report['best']=best; report['elapsed_s']=time.time()-started
    if best:
        report['activation_skipped']='Candidato sperimentale: serve un test finale separato prima della selezione. Nessuna attivazione automatica.'
    save_report()
    print('FINE',json.dumps({'activated':report['activated'],'baseline_pass':baseline['probe_pass'],'best_pass':best['probe_pass'] if best else None,'report':str(out/'report.json')}))


if __name__=='__main__': main()
