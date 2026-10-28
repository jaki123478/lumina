"""Microfono Vosk locale -> ricerca web -> riepilogo e voce Piper locali."""
import argparse,json,sys,wave
from pathlib import Path
import numpy as np
import requests
from lumina_voice_neural import synthesize,play
ROOT=Path(__file__).resolve().parent
_recognizer_model=None

def transcribe(samples,rate,with_info=False):
    global _recognizer_model
    from vosk import Model,KaldiRecognizer,SetLogLevel
    from scipy.signal import resample_poly
    from math import gcd
    SetLogLevel(-1)
    if rate!=16000:
        g=gcd(rate,16000);samples=resample_poly(samples.astype(float),16000//g,rate//g)
    samples=np.clip(samples,-32768,32767).astype('<i2')
    if _recognizer_model is None:_recognizer_model=Model(str(ROOT/'voices/vosk-it'))
    recognizer=KaldiRecognizer(_recognizer_model,16000)
    recognizer.SetWords(True);segments=[]
    for i in range(0,len(samples),4000):
        if recognizer.AcceptWaveform(samples[i:i+4000].tobytes()):segments.append(json.loads(recognizer.Result()))
    segments.append(json.loads(recognizer.FinalResult()))
    text=' '.join(s.get('text','') for s in segments).strip()
    confidences=[w['conf'] for s in segments for w in s.get('result',[]) if 'conf' in w]
    info={'text':text,'confidence':sum(confidences)/len(confidences) if confidences else 0.,'segments':len(segments)}
    return info if with_info else text

def needs_confirmation(text,confidence):
    has_intent=any(w in text.lower().split() for w in ['cerca','cercami','trova','notizie','ricerca','spiegami','come','chi','quali','quando','dove','perché'])
    return confidence<.8 or len(text.split())<2 or (len(text.split())>=10 and not has_intent)

def speak(text,path):
    synthesize(text,path);play(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--query');p.add_argument('--audio-input');p.add_argument('--no-play',action='store_true');args=p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    folder=ROOT/'audio_neurale/ricerche_online';folder.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    if not args.query and not args.audio_input:
        print('Lumina: ricerca online con riconoscimento e voce locali. Il microfono si attiva soltanto premendo INVIO.',flush=True)
        speak('Sono Lumina. Premi invio, poi pronuncia la tua ricerca. Il microfono resta attivo per dieci secondi. Puoi dire esci per terminare.',folder/'istruzioni.wav')
    while True:
        if args.query:query=args.query
        elif args.audio_input:
            with wave.open(args.audio_input) as w:
                if w.getsampwidth()!=2 or w.getnchannels()!=1:raise ValueError('Serve un WAV mono PCM16')
                rate=w.getframerate();samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
            query=transcribe(samples,rate)
        else:
            choice=input('INVIO per parlare per 10 secondi, oppure scrivi esci > ').strip()
            if choice.lower()=='esci':break
            import sounddevice as sd
            print('MICROFONO ATTIVO: fai la tua domanda adesso.',flush=True)
            samples=sd.rec(160000,samplerate=16000,channels=1,dtype='int16');sd.wait()
            info=transcribe(samples.reshape(-1),16000,with_info=True);query=info['text']
            print('Trascrizione:',query,'| affidabilità:',round(info['confidence'],2),flush=True)
            if query and needs_confirmation(query,info['confidence']):
                speak('La domanda riconosciuta non è abbastanza chiara. Controlla il testo nella console, poi conferma o ripeti.',folder/'conferma.wav')
                confirm=input('INVIO per confermare, r per ripetere, oppure digita la domanda corretta > ').strip()
                if confirm.lower()=='r':continue
                if confirm:query=confirm
        print('Domanda riconosciuta:',query,flush=True)
        if query.lower() in ('esci','fermati','basta'):break
        if not query:
            print('Nessuna domanda riconosciuta.')
            if args.query or args.audio_input:break
            continue
        if not args.no_play:speak('Sto cercando online e consultando le fonti. La tua voce rimane sul computer.',folder/'attesa.wav')
        r=session.post('http://127.0.0.1:8000/v1/chat/completions',json={'model':'lumina-assisted','messages':[{'role':'user','content':query}],'research':True,'temperature':0},timeout=240);r.raise_for_status();data=r.json()
        print(data['choices'][0]['message']['content'],flush=True)
        meta=data.get('backend',{});summary=meta.get('spoken_summary',data['choices'][0]['message']['content'])
        sources=meta.get('sources',[])
        text=summary+(f' Le pagine consultate sono {len(sources)}; i collegamenti sono nel registro locale.' if sources else '')
        path=synthesize(text,folder/'risposta.wav')
        (folder/'ricerca.json').write_text(json.dumps({'query':query,'result':data},ensure_ascii=False,indent=2),encoding='utf-8')
        print('Audio:',path,flush=True)
        if not args.no_play:play(path)
        if args.query or args.audio_input:break

if __name__=='__main__':main()
