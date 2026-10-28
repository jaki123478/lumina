"""Legge esercizi e avanzamenti dagli altoparlanti durante un ciclo locale."""
import argparse,json,time,sys
from pathlib import Path
from lumina_voice_neural import synthesize,play,clean
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--log',required=True);args=p.parse_args()
    run=(ROOT/args.run).resolve();log=(ROOT/args.log).resolve()
    if not run.is_relative_to(ROOT) or not log.is_relative_to(ROOT):raise ValueError('Percorsi esterni al progetto')
    deadline=time.time()+90
    while not run.exists() and time.time()<deadline:
        time.sleep(.5)
    if not run.exists():raise RuntimeError('La cartella del ciclo non è stata creata')
    audio=run/'voce_live';audio.mkdir(exist_ok=True)
    stop=run/'FERMA_VOCE';index=0;last_progress='';started=time.time()
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    def say(text):
        path=synthesize(text,audio/'lettura_corrente.wav')
        print('VOCE:',clean(text),flush=True);play(path)
    say('Sono Lumina. Ti tengo compagnia a voce mentre prosegue il mio addestramento locale. Leggerò esercizi di riferimento e avanzamenti. Le mie risposte vengono ancora controllate.')
    while not stop.exists():
        text=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
        if 'Traceback (most recent call last)' in text:
            say('Il ciclo di addestramento si è fermato con un errore. Interrompo la lettura mentre viene controllato.');break
        if any(line.startswith(('FINE ','CICLO COMPLETATO.')) for line in text.splitlines()):
            say('Questo ciclo di addestramento è terminato. Ora vengono confrontati i risultati. Non considero corrette le risposte prima della verifica.');break
        progress=[line for line in text.splitlines() if line.startswith(('TRAIN ','CONTROLLO ','BASELINE '))]
        latest=progress[-1] if progress else ''
        if latest!=last_progress and latest.startswith('TRAIN '):
            step=latest.split()[1].rstrip(':');say('Avanzamento: aggiornamento '+step.replace('/',' di ')+'. Continuo con gli esercizi controllati.');last_progress=latest
        rows=[];source=run/'train.jsonl'
        if source.exists():
            for line in source.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row.get('category') in ('conoscenze','estrazione','memoria') and '```' not in row['assistant']:
                    rows.append(row)
        if rows:
            row=rows[index%len(rows)];index+=1
            history=[m['content'] for m in row['messages'][:-1] if m['role']=='user']
            question=row['messages'][-1]['content']
            say('Esercizio di riferimento. '+' '.join(history+[question])+' Risposta di riferimento: '+row['assistant'])
        else:
            say('Sto preparando le lezioni e confrontando le risposte iniziali. Tutto avviene sul computer. Attendo che gli aggiornamenti siano pronti.')
        time.sleep(.6)
        if time.time()-started>7200:
            say('Interrompo la lettura automatica dopo due ore. Il registro conserva lo stato del ciclo.');break
    print('VOCE TERMINATA',flush=True)


if __name__=='__main__':main()
