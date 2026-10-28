import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../dist/local-voice.js',import.meta.url),'utf8');
const {LocalVoiceSynth}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
let calls=0,played=0,paused=0;
globalThis.fetch=async (url,options)=>{
 assert.equal(url,'/api/tts');assert.equal(options.method,'POST');
 assert.equal(JSON.parse(options.body).text,'Ciao');calls++;
 return {ok:true,blob:async()=>new Blob(['RIFF'],{type:'audio/wav'})};
};
globalThis.Audio=class {
 async play(){played++;queueMicrotask(()=>this.onended());}
 pause(){paused++;}
};
const voice=new LocalVoiceSynth();
await voice.speak('Ciao');
assert.equal(calls,1);assert.equal(played,1);assert.equal(voice.audio,null);
globalThis.fetch=async()=>({ok:false});
await assert.rejects(voice.speak('Ciao'),/locale/);
globalThis.fetch=async (_url,options)=>new Promise((_resolve,reject)=>{
 options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('abort'),{name:'AbortError'})));
});
const pending=voice.speak('Ciao');voice.stop();await pending;
assert.equal(voice.controller,null);
console.log('LocalVoiceSynth: riproduzione, errore server e cancellazione superati.');
