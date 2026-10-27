/** Lumina Voice 1: original Italian grapheme-to-phoneme + formant synthesizer.
 * No recordings, browser speech service, pretrained weights, or network calls.
 * Intentionally experimental and robotic; this is not a neural human voice.
 */
const VOWELS = {
  a: [730, 1090, 2440], e: [500, 1750, 2500], i: [270, 2290, 3010],
  o: [570, 840, 2410], u: [300, 870, 2240],
};
const SONORANTS = {
  m: [250, 1000, 2200], n: [280, 1500, 2500], ny: [300, 1950, 2700],
  ng: [260, 1300, 2400], l: [400, 1200, 2600], ly: [300, 1950, 2900],
  r: [400, 1300, 1700], j: [270, 2200, 3000], w: [320, 800, 2200],
};
const DIGITS = ['zero','uno','due','tre','quattro','cinque','sei','sette','otto','nove'];
const TEENS = ['dieci','undici','dodici','tredici','quattordici','quindici','sedici','diciassette','diciotto','diciannove'];
const TENS = ['','','venti','trenta','quaranta','cinquanta','sessanta','settanta','ottanta','novanta'];
function numberWord(n) {
  if(n < 10) return DIGITS[n];
  if(n < 20) return TEENS[n-10];
  if(n < 100) return TENS[Math.floor(n/10)].slice(0, n%10===1 || n%10===8 ? -1 : undefined)+(n%10 ? DIGITS[n%10] : '');
  if(n < 1000) return (n < 200 ? 'cento' : DIGITS[Math.floor(n/100)]+'cento')+(n%100 ? numberWord(n%100) : '');
  if(n < 1000000) return (n < 2000 ? 'mille' : numberWord(Math.floor(n/1000))+'mila')+(n%1000 ? numberWord(n%1000) : '');
  return String(n).split('').map(c=>DIGITS[Number(c)]).join(' ');
}

export function phonemize(text) {
  const normalized = String(text).slice(0,3000).toLowerCase()
    .replace(/\bllm\b/g,'elle elle emme').replace(/\bai\b/g,'a i')
    .replace(/\d[\d.]*/g, s=>numberWord(Number(s.replaceAll('.',''))))
    .normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/[’']/g,' ');
  const out = [];
  for(const word of normalized.match(/[a-z]+|[.,!?;:\n]/g)||[]) {
    if(!/^[a-z]/.test(word)) {out.push(word===',' || word===':' || word===';' ? ',' : '.');continue;}
    for(let i=0;i<word.length;i++) {
      const c=word[i], n=word[i+1]||'', nn=word[i+2]||'';
      if(c==='s' && n==='c' && /[ie]/.test(nn) && nn) {
        out.push('sh'); i+=2;
        if(nn==='e' || !/[aou]/.test(word[i+1]||'!')) out.push(nn);
        continue;
      }
      if(c==='g' && n==='n') {out.push('ny');i++;continue;}
      if(c==='g' && n==='l' && nn==='i') {out.push('ly');i+=2;if(!/[aeou]/.test(word[i+1]||'!'))out.push('i');continue;}
      if(c==='c' || c==='g') {
        if(n==='h') {out.push(c==='c'?'k':'g');i++;continue;}
        if(n==='e' || n==='i') {
          out.push(c==='c'?'ch':'jh');
          if(n==='i' && /[aou]/.test(nn) && nn)i++;
        } else out.push(c==='c'?'k':'g');
        continue;
      }
      if(c==='q') {out.push('k');if(n==='u'){out.push('w');i++;}continue;}
      if(c==='h')continue;
      if(c==='x'){out.push('k','s');continue;}
      if(c==='z'){out.push('ts');continue;}
      if(c==='y'){out.push('i');continue;}
      if(c==='s' && i>0 && VOWELS[word[i-1]] && VOWELS[n]){out.push('z');continue;}
      if(c==='n' && /[cgkq]/.test(n) && n){out.push('ng');continue;}
      out.push(c);
    }
    out.push(' ');
  }
  return out;
}

function resonator(freq, bandwidth, sampleRate) {
  const r=Math.exp(-Math.PI*bandwidth/sampleRate);
  return {a:2*r*Math.cos(2*Math.PI*freq/sampleRate),b:-r*r,g:1-r,y1:0,y2:0};
}
function filter(s,x) {const y=s.g*x+s.a*s.y1+s.b*s.y2;s.y2=s.y1;s.y1=y;return y;}

/** Deterministic source-filter synthesis. All synthesis rules live in this file. */
export function synthesize(text,{sampleRate=22050,speed=1,pitch=120}={}) {
  speed=Math.max(0.6,Math.min(1.6,speed));
  pitch=Math.max(80,Math.min(220,pitch));
  const phonemes=phonemize(text);
  const segments=phonemes.map((p,i)=>{
    let duration=p==='.'?0.25:p===','?0.12:p===' '?0.035:VOWELS[p]?0.125:SONORANTS[p]?0.075:0.075;
    if(phonemes[i+1]===p && p!==' ')duration*=1.1;
    return {p,n:Math.floor(sampleRate*duration/speed)};
  });
  const result=new Float32Array(segments.reduce((n,s)=>n+s.n,0));
  let offset=0,phase=0,seed=9417;
  const noise=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/2147483648-1;};
  for(let si=0;si<segments.length;si++) {
    const {p,n}=segments[si];
    if(p===' ' || p==='.' || p===','){offset+=n;continue;}
    const isVowel=Boolean(VOWELS[p]),isSonorant=Boolean(SONORANTS[p]);
    const stop=/^(p|b|t|d|k|g|ch|jh|ts)$/.test(p);
    const fricative=/^(f|v|s|z|sh|ch|jh|ts)$/.test(p);
    const voiced=isVowel||isSonorant||/^(b|d|g|v|z|jh)$/.test(p);
    const nextVowel=phonemes.slice(si+1,si+4).find(x=>VOWELS[x]);
    const formants=VOWELS[p]||SONORANTS[p]||VOWELS[nextVowel]||VOWELS.e;
    const filters=formants.map((f,i)=>resonator(f,[100,150,220][i],sampleRate));
    const noiseBand=resonator(p==='sh'||p==='ch'||p==='jh'?2800:p==='f'||p==='v'?4200:5700,1700,sampleRate);
    let prevNoise=0, previousGlottal=0;
    for(let j=0;j<n;j++) {
      const t=j/sampleRate,u=j/n;
      const localPitch=pitch*(1+0.018*Math.sin(2*Math.PI*5.2*(offset+j)/sampleRate))*(1-0.06*si/Math.max(1,segments.length));
      phase+=localPitch/sampleRate;phase-=Math.floor(phase);
      const glottal=phase<0.42 ? 0.5*(1-Math.cos(2*Math.PI*phase/0.42)) : -0.13*Math.sin(Math.PI*(phase-0.42)/0.58);
      const excitation=(glottal-previousGlottal)*6;previousGlottal=glottal;
      let sample=0;
      if(voiced)sample=(filter(filters[0],excitation)+0.7*filter(filters[1],excitation)+0.35*filter(filters[2],excitation))*1.3;
      if(p==='r')sample*=0.35+0.65*Math.max(0,Math.sin(2*Math.PI*27*t));
      if(p==='m'||p==='n'||p==='ng'||p==='ny')sample*=0.65;
      const white=noise(),high=white-prevNoise*0.85;prevNoise=white;
      if(fricative)sample+=filter(noiseBand,high)*(p==='f'||p==='v'?0.65:1.6);
      if(stop) {
        const release=u>0.47 && u<0.6;
        sample=(u<0.45 ? sample*0.07 : sample)+(release?high*0.55*(1-(u-0.47)/0.13):0);
      }
      if(!voiced && !fricative && !stop)sample=high*0.05;
      const env=Math.min(1,j/(sampleRate*0.008),(n-j)/(sampleRate*0.012));
      result[offset+j]=Math.tanh(sample)*Math.max(0,env)*0.7;
    }
    offset+=n;
  }
  return result;
}

export class VoiceSynth {
  constructor(){this.context=null;this.source=null;this.speaking=false;this._finish=null;this._request=0;}
  async unlock(){
    const AudioContext=globalThis.AudioContext||globalThis.webkitAudioContext;
    if(!AudioContext)throw new Error('Questo browser non supporta l’audio necessario alla voce Lumina.');
    this.context ||= new AudioContext();
    await this.context.resume();
  }
  async speak(text) {
    this.stop();
    const request=++this._request;
    await this.unlock();
    if(request!==this._request)return;
    const samples=synthesize(text);
    if(!samples.length)return;
    const buffer=this.context.createBuffer(1,samples.length,22050);
    buffer.copyToChannel(samples,0);
    const source=this.context.createBufferSource();source.buffer=buffer;source.connect(this.context.destination);
    this.source=source;this.speaking=true;
    return new Promise(resolve=>{
      const finish=()=>{if(this.source===source){this.source=null;this.speaking=false;}source.disconnect();if(this._finish===finish)this._finish=null;resolve();};
      this._finish=finish;source.onended=finish;source.start();
    });
  }
  stop(){this._request++;const s=this.source;this.source=null;this.speaking=false;if(s){s.onended=null;try{s.stop();}catch{} }const finish=this._finish;this._finish=null;finish?.();}
  async close(){this.stop();if(this.context){await this.context.close();this.context=null;}}
}
