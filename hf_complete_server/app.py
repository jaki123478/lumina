import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

MODEL_REPO = os.getenv('LUMINA_MODEL_REPO', 'jacopof/lumina-complete')
ROOT = Path('/tmp/lumina-model')
ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Lumina Complete API')
runtime = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = 'lumina-complete'
    messages: list[Message]
    max_tokens: int = 96

def get_runtime():
    global runtime
    if runtime is None:
        from lumina_onnx_runtime import LuminaOnnxRuntime
        model = hf_hub_download(MODEL_REPO, 'lumina_next_135m.onnx', local_dir=ROOT)
        vocab = hf_hub_download(MODEL_REPO, 'vocab.json', subfolder='tokenizer_lumina_next', local_dir=ROOT)
        merges = hf_hub_download(MODEL_REPO, 'merges.txt', subfolder='tokenizer_lumina_next', local_dir=ROOT)
        runtime = LuminaOnnxRuntime(model, vocab, merges)
    return runtime

@app.get('/health')
def health():
    return {'status':'ok', 'model':'lumina-complete'}

@app.get('/v1/models')
def models():
    return {'object':'list','data':[{'id':'lumina-complete','object':'model','owned_by':'jacopof'}]}

@app.post('/v1/chat/completions')
def chat(req: ChatRequest):
    prompt = req.messages[-1].content if req.messages else ''
    text = get_runtime().generate(prompt, max_new_tokens=min(max(req.max_tokens,1),192))
    return {'id':'lumina-complete','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop'}]}
