import { LocalVoiceSynth as VoiceSynth } from './local-voice.js';

const $ = id => document.getElementById(id);
const iconPaths = {
  plus: '<path d="M12 5v14M5 12h14"/>', chat: '<path d="M20 11.5a8 8 0 0 1-8 8H5l-3 2v-10a9 9 0 0 1 18 0Z"/><path d="M7 10h8M7 14h5"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/><rect x="10" y="10" width="4" height="4" rx=".5"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>', menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  sliders: '<path d="M4 7h6M14 7h6M4 17h10M18 17h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  spark: '<path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3Z"/>',
  book: '<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15"/>',
  wave: '<path d="M3 10v4M7 6v12M12 3v18M17 7v10M21 10v4"/>',
  'arrow-up': '<path d="M12 19V5m-6 6 6-6 6 6"/>', stop: '<rect x="7" y="7" width="10" height="10" rx="1" fill="currentColor" stroke="none"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>', mic: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
  phone: '<path d="M5 3h4l2 5-3 2a16 16 0 0 0 6 6l2-3 5 2v4a2 2 0 0 1-2 2C10 21 3 14 3 5a2 2 0 0 1 2-2Z"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/>',
  copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  paperclip: '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.49-8.49a4 4 0 0 1 5.66 5.66l-8.49 8.49a2 2 0 0 1-2.83-2.83l7.78-7.78"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${iconPaths[name] || iconPaths.spark}</svg>`;
function setIcon(el, name) { el.innerHTML = icon(name); }
document.querySelectorAll('[data-icon]').forEach(el => setIcon(el, el.dataset.icon));

const STORAGE = 'lumina.chats.v1';
const SETTINGS_STORAGE = 'lumina.settings.v1';
let conversations = [];

let activeDocument = null;

function setupRAG() {
  const uploadBtn = $('upload-doc-btn');
  const docInput = $('doc-input');

  if (uploadBtn && docInput) {
    uploadBtn.addEventListener('click', () => docInput.click());
    docInput.addEventListener('change', async () => {
      const file = docInput.files?.[0];
      if (file) await processUploadedFile(file);
    });
  }

  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', async e => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) await processUploadedFile(file);
  });
}

async function processUploadedFile(file) {
  showNotice(`Caricamento ed estrazione di ${file.name}...`);
  try {
    let payload = { filename: file.name };
    if (file.name.toLowerCase().endsWith('.pdf')) {
      const arrayBuffer = await file.arrayBuffer();
      let binary = '';
      const bytes = new Uint8Array(arrayBuffer);
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      payload.content_b64 = btoa(binary);
    } else {
      payload.text = await file.text();
    }

    const res = await fetch('/api/parse_doc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === 'ok') {
      activeDocument = data;
      renderDocChip();
      showNotice(`Documento indicizzato: ${data.chunks_count} sezioni pronte.`);
      setTimeout(() => showNotice(''), 3000);
    } else {
      showNotice(`Errore documento: ${data.message || 'impossibile leggere il file'}`);
    }
  } catch (err) {
    showNotice(`Errore caricamento documento: ${err.message}`);
  }
}

function renderDocChip() {
  const container = $('doc-chip-container');
  if (!container) return;
  if (!activeDocument) {
    container.style.display = 'none';
    container.innerHTML = '';
    return;
  }
  container.style.display = 'block';
  container.innerHTML = `
    <div class="doc-chip">
      <span>📄 ${activeDocument.filename} (${activeDocument.chunks_count} sezioni · ${activeDocument.total_words} parole)</span>
      <button type="button" id="remove-doc-btn" aria-label="Rimuovi documento">×</button>
    </div>
  `;
  $('remove-doc-btn')?.addEventListener('click', () => {
    activeDocument = null;
    renderDocChip();
    const docInput = $('doc-input');
    if (docInput) docInput.value = '';
    showNotice('Documento rimosso.');
    setTimeout(() => showNotice(''), 2000);
  });
}

let selectedId = null;
let loaded = false;
let active = null;
let worker;
let settings = { temperature: 0.25, maxTokens: 256 };
const voice = { live: false, recording: false, speaking: false, preview: false, epoch: 0, recognizer: null, synth: new VoiceSynth(), phrases: ['Ciao', 'Come ti chiami?', 'Come funzioni?', 'Racconta una storia'] };

try {
  const stored = JSON.parse(localStorage.getItem(STORAGE) || '[]');
  if (Array.isArray(stored)) conversations = stored.slice(0, 100).filter(c => c && typeof c.id === 'string' && Array.isArray(c.messages)).map(c => ({ id: c.id, title: typeof c.title === 'string' ? c.title.slice(0, 80) : 'Conversazione', updatedAt: Number(c.updatedAt) || 0, messages: c.messages.slice(-100).filter(m => m && ['user', 'assistant'].includes(m.role) && typeof m.content === 'string' && (m.role !== 'assistant' || m.content.trim())).map(m => ({ role: m.role, content: m.content.slice(0, 10000), tokens: m.tokens, durationMs: m.durationMs })) }));
  const saved = JSON.parse(localStorage.getItem(SETTINGS_STORAGE) || '{}');
  if (Number.isFinite(saved.temperature)) settings.temperature = Math.max(0.1, Math.min(1.1, saved.temperature));
  if ([128, 256, 384].includes(saved.maxTokens)) settings.maxTokens = saved.maxTokens;
} catch { /* A damaged browser cache should never prevent a fresh chat. */ }

function uid() { return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`; }
function currentChat() { return conversations.find(c => c.id === selectedId); }
function saveChats() {
  try { localStorage.setItem(STORAGE, JSON.stringify(conversations.slice(0, 100))); }
  catch { showNotice('Il browser non può salvare altre chat. Puoi eliminare la cronologia dalle impostazioni.'); }
}
function saveSettings() { try { localStorage.setItem(SETTINGS_STORAGE, JSON.stringify(settings)); } catch {} }
function showNotice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function scrollDown() { $('conversation-scroll').scrollTop = $('conversation-scroll').scrollHeight; }
function shouldScroll() { const el = $('conversation-scroll'); return el.scrollHeight - el.scrollTop - el.clientHeight < 100; }

function closeSidebar() { $('sidebar').classList.remove('open'); $('mobile-scrim').hidden = true; $('mobile-menu').setAttribute('aria-expanded', 'false'); }
function renderHistory() {
  const list = $('chat-history'); list.replaceChildren();
  $('history-count').textContent = conversations.length; $('history-empty').hidden = conversations.length > 0;
  [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).forEach(chat => {
    const row = document.createElement('div'); row.className = `chat-row${chat.id === selectedId ? ' active' : ''}`;
    const button = document.createElement('button'); button.className = 'chat-link'; button.textContent = chat.title; button.title = chat.title;
    if (chat.id === selectedId) button.setAttribute('aria-current', 'page');
    button.addEventListener('click', () => { if (active) stopGeneration(); selectedId = chat.id; renderHistory(); renderConversation(); closeSidebar(); });
    const remove = document.createElement('button'); remove.className = 'chat-delete'; remove.setAttribute('aria-label', `Elimina ${chat.title}`); remove.innerHTML = `<span class="icon">${icon('trash')}</span>`;
    remove.addEventListener('click', () => {
      if (chat.id === selectedId && active) stopGeneration(); conversations = conversations.filter(c => c.id !== chat.id);
      if (selectedId === chat.id) selectedId = null;
      saveChats(); renderHistory(); renderConversation();
    });
    row.append(button, remove); list.append(row);
  });
}

function createMessage(message) {
  const article = document.createElement('article'); article.className = `message ${message.role}`;
  if (message.role === 'assistant') {
    const avatar = document.createElement('span'); avatar.className = 'message-avatar'; avatar.setAttribute('aria-hidden', 'true');
    avatar.innerHTML = '<svg viewBox="0 0 48 48"><path d="M24 6v36M6 24h36M11.3 11.3l25.4 25.4M11.3 36.7l25.4-25.4"/></svg>'; article.append(avatar);
  }
  const body = document.createElement('div'); body.className = 'message-body';
  const author = document.createElement('div'); author.className = 'message-author'; author.textContent = message.role === 'user' ? 'Tu' : 'Lumina';
  const text = document.createElement('div'); text.className = 'message-text'; text.textContent = message.content; body.append(author, text);
  if (message.role === 'assistant') {
    const meta = document.createElement('div'); meta.className = 'message-meta';
    if (message.content) {
      const copy = document.createElement('button'); copy.className = 'copy-button'; copy.setAttribute('aria-label', 'Copia risposta'); copy.innerHTML = `<span class="icon">${icon('copy')}</span>`;
      copy.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(message.content); copy.innerHTML = `<span class="icon">${icon('check')}</span>`; setTimeout(() => { copy.innerHTML = `<span class="icon">${icon('copy')}</span>`; }, 1500); }
        catch { showNotice('Copia non disponibile: seleziona il testo della risposta.'); }
      }); meta.append(copy);
    }
    if (message.tokens) { const detail = document.createElement('span'); detail.textContent = `${message.tokens} token${message.durationMs ? ` · ${(message.durationMs / 1000).toLocaleString('it-IT', { maximumFractionDigits: 1 })} s` : ''}`; meta.append(detail); }
    body.append(meta);
  }
  article.append(body); return article;
}

function renderConversation() {
  const chat = currentChat(); const hasMessages = !!chat?.messages.length;
  $('welcome').hidden = hasMessages; $('messages').hidden = !hasMessages; $('messages').replaceChildren();
  if (chat) chat.messages.forEach(m => $('messages').append(createMessage(m)));
  if (active && chat?.id === active.chat.id) {
    const article = $('messages').lastElementChild; article.setAttribute('aria-busy', 'true'); active.element = article.querySelector('.message-text');
    if (!active.response.content) active.element.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  }
  scrollDown(); updateComposer();
}
function updateComposer() {
  const button = $('send-button'); button.disabled = !active && (!loaded || !$('prompt').value.trim()); button.classList.toggle('busy', !!active);
  button.setAttribute('aria-label', active ? 'Interrompi risposta' : 'Invia messaggio'); button.title = active ? 'Interrompi risposta' : 'Invia messaggio';
  setIcon(button.querySelector('.icon'), active ? 'stop' : 'arrow-up');
}
function resizePrompt() { $('prompt').style.height = 'auto'; $('prompt').style.height = `${Math.min(160, $('prompt').scrollHeight)}px`; updateComposer(); }

function sendMessage(content, fromVoice = false) {
  const text = content.trim(); if (!text || active) return;
  if (!loaded) { showNotice('Attendi che il modello sia pronto prima di inviare.'); return; }
  showNotice(''); let chat = currentChat();
  if (!chat) { chat = { id: uid(), title: text.slice(0, 44), updatedAt: Date.now(), messages: [] }; selectedId = chat.id; conversations.unshift(chat); }
  chat.messages.push({ role: 'user', content: text });
  const response = { role: 'assistant', content: '' }; chat.messages.push(response); chat.updatedAt = Date.now();
  const id = uid(); active = { id, chat, response, fromVoice };
  $('prompt').value = ''; resizePrompt(); renderHistory(); renderConversation();
  const article = $('messages').lastElementChild; article.setAttribute('aria-busy', 'true');
  article.querySelector('.message-text').innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>'; active.element = article.querySelector('.message-text');
  saveChats(); 
  // 1. Gestione Documento RAG (se allegato)
  if (activeDocument) {
    const selectedModel = $('model-selector')?.value || 'lumina-coder';
    fetch('/api/rag_chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: activeDocument.filename,
        chunks: activeDocument.chunks,
        question: text,
        model: selectedModel
      })
    })
    .then(r => r.json())
    .then(data => {
      const run = active; active = null;
      let ans = data.answer || "Nessuna informazione rilevante trovata nel documento.";
      if (data.source_chunk) {
        ans += `\n\n> 📄 **Fonte verificata da ${data.document}**:\n> *"${data.source_chunk.trim()}"*`;
      }
      run.response.content = ans;
      saveChats();
      renderConversation();
      updateComposer();
      if (fromVoice && voice.live) speakReply(run.response.content);
    })
    .catch(err => {
      const run = active; active = null;
      run.response.content = `Errore RAG: ${err.message}. Verifica che il server locale sia attivo.`;
      saveChats();
      renderConversation();
      updateComposer();
    });
    return;
  }

  // 2. Gestione Selezione Modello
  const selectedModel = $('model-selector')?.value || 'auto';
  if (selectedModel !== 'lumina-50k') {
    fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel === 'auto' ? 'lumina-assisted' : selectedModel,
        messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
        max_tokens: settings.maxTokens || 250,
        temperature: settings.temperature ?? 0.2
      })
    })
    .then(r => r.json())
    .then(data => {
      const run = active; active = null;
      const reply = data.choices?.[0]?.message?.content || "";
      run.response.content = reply;
      saveChats();
      renderConversation();
      updateComposer();
      if (fromVoice && voice.live) speakReply(run.response.content);
    })
    .catch(() => {
      const run = active; active = null;
      run.response.content = 'Server Lumina locale non disponibile. Avvialo oppure scegli Lumina Mini nel menu.';
      saveChats(); renderConversation(); updateComposer();
    });
    return;
  }

  worker.postMessage({ type: 'generate', id, messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })), maxTokens: settings.maxTokens, temperature: settings.temperature });

  if (fromVoice) { setVoiceState('thinking', 'Sto preparando una risposta…'); $('voice-transcript').textContent = `Tu: ${text}`; }
}
function stopGeneration() {
  if (!active) return; const run = active; worker.postMessage({ type: 'cancel', id: run.id }); active = null;
  if (!run.response.content) run.chat.messages = run.chat.messages.filter(m => m !== run.response);
  saveChats(); renderConversation(); updateComposer();
  if (run.fromVoice && voice.live) { voice.live = false; setVoiceState('idle', 'Risposta interrotta. Premi il microfono per riprendere.'); }
}
function setStatus(state, text) { $('model-status').className = `model-status ${state}`; $('status-text').textContent = text; $('model-state-detail').textContent = state === 'ready' ? 'Pronto' : state === 'error' ? 'Da verificare' : 'Caricamento…'; }

try {
  worker = new Worker('./inference-worker.js');
  worker.addEventListener('message', event => {
    const message = event.data;
    if (message.type === 'progress') { setStatus('loading', `Caricamento ${Math.round(message.progress * 100)}%`); return; }
    if (message.type === 'ready') { loaded = true; setStatus('ready', 'Pronto · tutto locale'); updateComposer(); $('parameter-count').textContent = 'Qwen locale · Next · Mini'; return; }
    if (message.type === 'error') {
      if (message.id && active?.id !== message.id) return;
      if (!message.id) { loaded = false; setStatus('error', 'Modello non disponibile'); }
      showNotice(message.message || 'Il modello non è disponibile. Ricarica la pagina.');
      if (active) { const run = active; active = null; if (!run.response.content) run.chat.messages = run.chat.messages.filter(m => m !== run.response); saveChats(); renderConversation(); }
      if (voice.live) { voice.live = false; setVoiceState('idle', 'Risposta non disponibile. Controlla la chat.'); } updateComposer(); return;
    }
    if (active?.id !== message.id) return;
    if (message.type === 'token') {
      const scroll = shouldScroll(); active.response.content += message.text; active.element.textContent = active.response.content;
      const cursor = document.createElement('span'); cursor.className = 'generation-cursor'; cursor.setAttribute('aria-hidden', 'true'); active.element.append(cursor); if (scroll) scrollDown();
    }
    if (message.type === 'done') {
      const run = active; active = null; run.response.content = message.text || run.response.content; run.response.tokens = message.tokens; run.response.durationMs = message.durationMs;
      if (!run.response.content) { run.chat.messages = run.chat.messages.filter(m => m !== run.response); showNotice('Il modello ha terminato senza testo. Prova una domanda più breve o aumenta la creatività.'); }
      saveChats(); renderConversation(); updateComposer(); if (message.truncated) showNotice('Il messaggio supera il breve contesto del modello: Lumina ha letto solo la parte finale.');
      if (run.fromVoice && voice.live) speakReply(run.response.content);
    }
  });
  worker.addEventListener('error', () => { loaded = false; setStatus('error', 'Modello non disponibile'); if (active) stopGeneration(); showNotice('Impossibile avviare il modello. Apri l’app dal suo indirizzo web, poi ricarica la pagina.'); updateComposer(); }); worker.postMessage({ type: 'load' });
} catch (error) { setStatus('error', 'Browser non compatibile'); showNotice(`Impossibile avviare il modello: ${error.message}`); }

function setVoiceState(state, text) {
  $('voice-orb').className = `voice-orb ${state}`; $('voice-state').textContent = text;
  $('voice-title').textContent = state === 'listening' ? 'Ti ascolto.' : state === 'thinking' ? 'Un attimo per pensare.' : state === 'speaking' ? 'Ecco cosa penso.' : state === 'recording' ? 'Insegna la tua voce.' : 'La parola a te.';
  $('voice-control').classList.toggle('active', ['listening', 'speaking'].includes(state));
  $('voice-control').setAttribute('aria-label', state === 'listening' ? 'Ferma microfono' : state === 'speaking' ? 'Interrompi voce' : 'Avvia microfono'); setIcon($('voice-control').querySelector('.icon'), state === 'speaking' ? 'stop' : 'mic');
}
function updateTemplates(list) {
  const current = $('calibration-phrase').value || voice.phrases[0]; for (const template of list) if (!voice.phrases.includes(template.label)) voice.phrases.push(template.label);
  const total = list.reduce((n, t) => n + t.examples, 0); $('sample-count').textContent = `${total} ${total === 1 ? 'esempio' : 'esempi'}`; $('calibration-phrase').replaceChildren();
  voice.phrases.forEach(label => { const count = list.filter(t => t.label === label).reduce((n, t) => n + t.examples, 0); const option = document.createElement('option'); option.value = label; option.textContent = `${label} · ${count} ${count === 1 ? 'esempio' : 'esempi'}`; $('calibration-phrase').append(option); });
  $('calibration-phrase').value = current; $('remove-samples').disabled = !list.some(t => t.label === $('calibration-phrase').value);
}
function setupRecognizer() {
  if (voice.recognizer) return true;
  if (!window.LuminaRecognizer) { setVoiceState('idle', 'Il modulo vocale non è disponibile. Ricarica la pagina.'); return false; }
  voice.recognizer = new window.LuminaRecognizer({
    onStatus: ({ state, message }) => { if (!$('voice-dialog').open) return; if (voice.recording) { $('calibration-status').textContent = message; setVoiceState(state === 'processing' ? 'thinking' : 'recording', message); } else if (voice.live && !active && !voice.speaking) setVoiceState(state === 'processing' ? 'thinking' : state === 'listening' ? 'listening' : 'idle', message); },
    onResult: result => { if (!voice.live || voice.recording || active || voice.speaking || !$('voice-dialog').open) return; voice.recognizer.stop(); $('voice-transcript').textContent = `Riconosciuto: ${result.text}`; sendMessage(result.text, true); },
    onError: message => { voice.live = false; setVoiceState('idle', message); $('calibration-status').textContent = message; },
    onTemplates: list => updateTemplates(list)
  }); updateTemplates(voice.recognizer.listTemplates()); return true;
}
function microphoneAvailable() { if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { setVoiceState('idle', 'Il microfono richiede HTTPS oppure localhost e un browser recente.'); return false; } return true; }
async function unlockAudio() { if (typeof voice.synth.unlock === 'function') await voice.synth.unlock(); }
async function startListening() {
  if (!loaded) { setVoiceState('idle', 'Attendi che il modello sia pronto.'); return; }
  if (!setupRecognizer() || !microphoneAvailable()) return;
  if (!voice.recognizer.listTemplates().length) { $('voice-calibration').open = true; setVoiceState('idle', 'Prima registra una delle frasi qui sotto.'); return; }
  if (active) { setVoiceState('thinking', 'Attendi che Lumina termini la risposta in chat.'); return; }
  const epoch = ++voice.epoch; voice.live = true; voice.speaking = false;
  try { await unlockAudio(); if (!voice.live || voice.epoch !== epoch) return; $('voice-calibration').open = false; await voice.recognizer.start(); if (voice.epoch === epoch && voice.live) setVoiceState('listening', 'Pronuncia una delle frasi che hai registrato.'); }
  catch (error) { if (voice.epoch === epoch) { voice.live = false; setVoiceState('idle', error.message || 'Microfono non disponibile. Controlla il permesso nel browser.'); } }
}
async function speakReply(text) {
  const epoch = voice.epoch; if (!text) { if (voice.live) await startListening(); return; }
  voice.speaking = true; setVoiceState('speaking', 'Lumina sta leggendo. Premi il pulsante per interrompere.'); $('voice-transcript').textContent = text;
  try { await voice.synth.speak(text); } catch (error) { voice.live = false; setVoiceState('idle', `Audio non disponibile: ${error.message}`); } finally { voice.speaking = false; }
  if (voice.live && voice.epoch === epoch && $('voice-dialog').open) await startListening();
}
function stopVoice() {
  voice.live = false; voice.recording = false; voice.speaking = false; voice.preview = false; voice.epoch += 1; voice.recognizer?.stop(); voice.synth.stop(); if (active?.fromVoice) stopGeneration();
  $('voice-preview').querySelector('span:last-child').textContent = 'Ascolta la voce';
  $('record-sample').querySelector('span:last-child').textContent = 'Registra esempio'; $('record-sample').classList.remove('recording'); $('voice-control').disabled = false; setVoiceState('idle', 'Premi il microfono per iniziare.');
}
function openVoice() { $('voice-dialog').showModal(); setupRecognizer(); setVoiceState('idle', voice.recognizer?.listTemplates().length ? 'Premi il microfono e pronuncia una frase registrata.' : 'Prima registra le frasi che vuoi usare.'); }
function closeVoice() { stopVoice(); $('voice-dialog').close(); }

$('chat-form').addEventListener('submit', event => { event.preventDefault(); if (active) stopGeneration(); else sendMessage($('prompt').value); });
$('prompt').addEventListener('input', resizePrompt);
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!active) sendMessage($('prompt').value); } });
document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => { if (!loaded) { $('prompt').value = button.dataset.prompt; resizePrompt(); $('prompt').focus(); } else sendMessage(button.dataset.prompt); }));
$('new-chat').addEventListener('click', () => { if (active) stopGeneration(); selectedId = null; showNotice(''); $('prompt').value = ''; resizePrompt(); renderHistory(); renderConversation(); closeSidebar(); $('prompt').focus(); });
$('mobile-menu').addEventListener('click', () => { const open = !$('sidebar').classList.contains('open'); $('sidebar').classList.toggle('open', open); $('mobile-scrim').hidden = !open; $('mobile-menu').setAttribute('aria-expanded', String(open)); }); $('mobile-scrim').addEventListener('click', closeSidebar);
function openSettings() { $('clear-confirm').hidden = true; $('settings-dialog').showModal(); }
$('settings-button').addEventListener('click', openSettings); $('model-details').addEventListener('click', openSettings); $('close-settings').addEventListener('click', () => $('settings-dialog').close());
$('temperature').value = settings.temperature; $('temperature-output').textContent = settings.temperature.toLocaleString('it-IT', { minimumFractionDigits: 2 }); $('response-length').value = settings.maxTokens;
$('temperature').addEventListener('input', event => { settings.temperature = Number(event.target.value); $('temperature-output').textContent = settings.temperature.toLocaleString('it-IT', { minimumFractionDigits: 2 }); saveSettings(); });
$('response-length').addEventListener('change', event => { settings.maxTokens = Number(event.target.value); saveSettings(); });
$('clear-history').addEventListener('click', () => { $('clear-confirm').hidden = false; }); $('cancel-clear').addEventListener('click', () => { $('clear-confirm').hidden = true; });
$('confirm-clear').addEventListener('click', () => { if (active) stopGeneration(); conversations = []; selectedId = null; saveChats(); renderHistory(); renderConversation(); $('clear-confirm').hidden = true; $('settings-dialog').close(); });
$('voice-button').addEventListener('click', openVoice); $('close-voice').addEventListener('click', closeVoice); $('end-voice').addEventListener('click', closeVoice); $('voice-dialog').addEventListener('cancel', () => { stopVoice(); });
$('voice-control').addEventListener('click', async () => { if (voice.recording) return; if (voice.preview) { stopVoice(); } else if (voice.speaking) { voice.epoch += 1; voice.speaking = false; voice.synth.stop(); voice.live = false; await startListening(); } else if (voice.live) stopVoice(); else await startListening(); });
$('voice-preview').addEventListener('click', async () => {
  if (voice.preview) { stopVoice(); return; }
  stopVoice(); const epoch = voice.epoch; voice.preview = true; voice.speaking = true;
  $('voice-preview').querySelector('span:last-child').textContent = 'Interrompi anteprima';
  const text = 'Ciao, sono Lumina. Questa è la mia voce sperimentale.';
  $('voice-transcript').textContent = text; setVoiceState('speaking', 'Anteprima della voce italiana locale. Microfono spento.');
  try { await unlockAudio(); if (voice.epoch !== epoch) return; await voice.synth.speak(text); if (voice.epoch === epoch) { voice.preview = false; voice.speaking = false; $('voice-preview').querySelector('span:last-child').textContent = 'Ascolta la voce'; setVoiceState('idle', 'Anteprima terminata. Registra una frase per iniziare a parlare.'); } }
  catch (error) { if (voice.epoch === epoch) { voice.preview = false; voice.speaking = false; $('voice-preview').querySelector('span:last-child').textContent = 'Ascolta la voce'; setVoiceState('idle', error.message || 'Audio non disponibile in questo browser.'); } }
});
$('record-sample').addEventListener('click', async () => {
  if (voice.recording) { voice.recognizer?.stop(); return; }
  if (active) { $('calibration-status').textContent = 'Attendi che la risposta in chat sia terminata.'; return; }
  if (!setupRecognizer() || !microphoneAvailable()) return; stopVoice(); const epoch = voice.epoch; voice.recording = true; $('voice-control').disabled = true;
  $('record-sample').querySelector('span:last-child').textContent = 'Ferma registrazione'; $('record-sample').classList.add('recording'); const phrase = $('calibration-phrase').value; $('voice-transcript').textContent = `Pronuncia: «${phrase}»`;
  try { await unlockAudio(); if (voice.epoch !== epoch || !$('voice-dialog').open) return; await voice.recognizer.record(phrase); if (voice.epoch === epoch) { $('calibration-status').textContent = 'Esempio salvato. Registra un secondo esempio o prova il microfono.'; setVoiceState('idle', 'Esempio registrato. La frase è pronta da provare.'); } }
  catch (error) { if (voice.epoch === epoch) { $('calibration-status').textContent = error.message || 'Registrazione non riuscita. Riprova.'; setVoiceState('idle', error.message || 'Riprova a registrare la frase.'); } }
  finally { if (voice.epoch === epoch) { voice.recording = false; $('voice-control').disabled = false; $('record-sample').querySelector('span:last-child').textContent = 'Registra esempio'; $('record-sample').classList.remove('recording'); updateTemplates(voice.recognizer.listTemplates()); } }
});
$('remove-samples').addEventListener('click', () => { if (!setupRecognizer()) return; const label = $('calibration-phrase').value; voice.recognizer.listTemplates().filter(t => t.label === label).forEach(t => voice.recognizer.deleteTemplate(t.id)); $('calibration-status').textContent = 'Esempi della frase eliminati.'; });
$('calibration-phrase').addEventListener('change', () => { $('remove-samples').disabled = !voice.recognizer?.listTemplates().some(t => t.label === $('calibration-phrase').value); });
$('add-phrase-form').addEventListener('submit', event => { event.preventDefault(); const phrase = $('custom-phrase').value.trim(); if (!phrase) return; if (!voice.phrases.includes(phrase)) voice.phrases.push(phrase); updateTemplates(voice.recognizer?.listTemplates() || []); $('calibration-phrase').value = phrase; $('custom-phrase').value = ''; $('calibration-status').textContent = 'Ora registra uno o più esempi di questa frase.'; });
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeSidebar(); if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'n') { event.preventDefault(); $('new-chat').click(); } });
window.addEventListener('pagehide', () => { saveChats(); voice.recognizer?.destroy(); voice.synth.stop(); worker?.terminate(); }); document.addEventListener('visibilitychange', () => { if (document.hidden && (voice.live || voice.recording || voice.speaking)) stopVoice(); });
renderHistory(); renderConversation();

setupRAG();
