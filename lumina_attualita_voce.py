"""Scarica titoli ufficiali e li legge con Piper sul PC. Nessun training automatico."""
from pathlib import Path
import argparse,json,sys
from lumina_web_sources import latest
from lumina_voice_neural import synthesize,play
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--tema',default='');p.add_argument('--no-play',action='store_true');args=p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
    data=latest(args.tema)
    if not data['items']:raise RuntimeError('Nessun titolo verificabile disponibile: '+str(data['errors']))
    sentences=['Sono Lumina. Leggo titoli da fonti ufficiali, scaricati da Internet. La voce viene generata sul tuo computer.']
    for row in data['items']:
        year,month,day=row['published'][:10].split('-')
        sentences.append(f"Fonte {row['source']}. Pubblicato il {day} {month} {year}. Titolo: {row['title']}.")
    text=' '.join(sentences)
    out=ROOT/'audio_neurale/attualita_locale';out.mkdir(exist_ok=True)
    data['spoken_text']=text;data['voice_weights_retrained']=False
    (out/'fonti.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    path=synthesize(text,out/'attualita.wav');print(text,flush=True);print('Audio:',path,flush=True)
    if not args.no_play:play(path)

if __name__=='__main__':main()
