"""
update_app_js.py - Integra RAG documenti, Model Selector e PWA in dist/app.js
"""
from pathlib import Path

app_js_path = Path("dist/app.js")
content = app_js_path.read_text(encoding="utf-8")

# 1. Aggiungi icona paperclip in iconPaths
if "paperclip:" not in content:
    target = "  check: '<path d=\"m5 12 4 4L19 6\"/>'"
    replacement = "  check: '<path d=\"m5 12 4 4L19 6\"/>',\n  paperclip: '<path d=\"m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.49-8.49a4 4 0 0 1 5.66 5.66l-8.49 8.49a2 2 0 0 1-2.83-2.83l7.78-7.78\"/>'"
    content = content.replace(target, replacement)

# 2. Aggiungi variabile activeDocument e gestione file RAG
rag_logic = """
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
"""

if "let activeDocument = null;" not in content:
    target = "let conversations = [];"
    content = content.replace(target, "let conversations = [];\n" + rag_logic)

# 3. Modifica sendMessage per gestire activeDocument e model selector
old_send_dispatch = "worker.postMessage({ type: 'generate', id, messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })), maxTokens: settings.maxTokens, temperature: settings.temperature });"

new_send_dispatch = """
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
        ans += `\\n\\n> 📄 **Fonte verificata da ${data.document}**:\\n> *"${data.source_chunk.trim()}"*`;
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
  if (selectedModel !== 'auto' && selectedModel !== 'lumina-2.75m') {
    fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
        max_tokens: settings.maxTokens || 250,
        temperature: settings.temperature || 0.2
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
      worker.postMessage({ type: 'generate', id, messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })), maxTokens: settings.maxTokens, temperature: settings.temperature });
    });
    return;
  }

  worker.postMessage({ type: 'generate', id, messages: chat.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })), maxTokens: settings.maxTokens, temperature: settings.temperature });
"""

if old_send_dispatch in content:
    content = content.replace(old_send_dispatch, new_send_dispatch)

# 4. Inizializza RAG alla fine
if "setupRAG();" not in content:
    content += "\nsetupRAG();\n"

app_js_path.write_text(content, encoding="utf-8")
print("dist/app.js updated successfully!")
