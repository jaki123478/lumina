"""Supporto Qwen e calcolo locale. Il backend è dichiarato nelle risposte API."""
import ast
from fractions import Fraction
import os
from pathlib import Path
import re
import threading

for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key]='1'
ROOT = Path(__file__).resolve().parent
MODEL_ID='Qwen/Qwen2.5-Coder-1.5B-Instruct'
GGUF_CANDIDATES = (
    ROOT / 'models' / 'gemma-3-4b-it-clean' / 'gemma-3-4b-it-Q4_K_M.gguf',
    ROOT / 'models' / 'gemma-3-4b-it' / 'gemma-3-4b-it-Q4_K_M.gguf',
    ROOT / 'models' / 'gemma-3-4b-it-v2' / 'gemma-3-4b-it-Q4_K_M.gguf',
)
_lock=threading.RLock()
_gguf_lock=threading.RLock()
_model=None
_tokenizer=None
_gguf_model=None
_gguf_path=None
_onnx_runtime=None
_next_model=None
_next_tokenizer=None
SYSTEM=("Sei Lumina, assistente AI vocale e testuale in italiano sviluppata da Jacopo Forti ed eseguita localmente sul computer dell'utente. "
        "Rispondi sempre in italiano, direttamente alla domanda, senza preamboli inutili e senza usare emoji salvo richiesta esplicita. "
        "Per domande puntuali restituisci il dato richiesto in modo chiaro e conciso (2-4 frasi complete, salvo richieste di codice o approfondimento). "
        "Rispetta esattamente nomi, numeri, formato e numero di elementi richiesti. "
        "Usa tutti i messaggi precedenti per recuperare i dati della conversazione. Non inventare nomi, dati privati, eventi futuri o fonti. "
        "Per codice restituisci codice completo, corretto e ben formattato. Non dire mai di essere Gemma, Qwen o Google.")


def calculate(prompt):
    """Espressioni aritmetiche chiuse; niente nomi, chiamate o esecuzione Python."""
    text=prompt.lower().replace('×','*').replace('÷','/').replace('−','-')
    text=re.sub(r'\bper\b','*',text)
    text=text.replace('più','+').replace('meno','-').replace('diviso','/')
    match=re.search(r'\b(?:calcola|quanto fa|quanto è)\s+([0-9\s.()+*/%\-]+)',text)
    if not match:return None
    if match.end()<len(text) and not match.group(1)[-1].isspace() and (text[match.end()].isalnum() or text[match.end()]=='_'):
        return None
    raw=match.group(1).strip().rstrip('.').strip()
    if len(raw)>100 or not any(c in raw for c in '+-*/%'):return None
    try:
        tree=ast.parse(raw,mode='eval')
        if len(list(ast.walk(tree)))>48:return None
        def value(node):
            if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                if abs(node.value)>10**12:raise ValueError('Numero troppo grande')
                return Fraction(str(node.value))
            if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
                return (-1 if isinstance(node.op,ast.USub) else 1)*value(node.operand)
            if isinstance(node,ast.BinOp):
                a,b=value(node.left),value(node.right)
                if isinstance(node.op,ast.Add):v=a+b
                elif isinstance(node.op,ast.Sub):v=a-b
                elif isinstance(node.op,ast.Mult):v=a*b
                elif isinstance(node.op,ast.Div):v=a/b
                elif isinstance(node.op,ast.Mod):v=a%b
                elif isinstance(node.op,ast.Pow) and b.denominator==1 and abs(b)<=8:v=a**int(b)
                else:raise ValueError('Operazione non ammessa')
                if abs(v)>10**18:raise ValueError('Risultato troppo grande')
                return v
            raise ValueError('Espressione non ammessa')
        result=value(tree.body)
        if result.denominator==1:return str(result.numerator)
        return format(float(result),'.12g')
    except (SyntaxError,ValueError,TypeError,ZeroDivisionError,OverflowError):
        return None


def _sanitize_public_identity(text: str) -> str:
    out = str(text or '').strip()
    out = re.sub(r'\b(?:io\s+)?sono\s+gemma(?:\s*[- ]?\s*3)?\b', 'sono Lumina', out, flags=re.IGNORECASE)
    out = re.sub(r'\bmi\s+chiamo\s+gemma(?:\s*[- ]?\s*3)?\b', 'mi chiamo Lumina', out, flags=re.IGNORECASE)
    out = re.sub(r'\bgemma(?:\s*[- ]?\s*3)?\b', 'Lumina', out, flags=re.IGNORECASE)
    out = re.sub(
        r'\b(?:creato|creata|addestrato|addestrata|sviluppato|sviluppata|realizzato|realizzata)\s+da\s+google(?:\s+deepmind)?\b',
        'eseguita localmente sul tuo computer',
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r'\bqwen(?:\s*2(?:\.5)?(?:-coder)?(?:-instruct)?)?\b', 'modello locale', out, flags=re.IGNORECASE)
    return out


def _trim_unfinished_sentence(answer: str, finish_reason: str | None) -> str:
    text = (answer or '').strip()
    if not text or finish_reason != 'length':
        return text
    if text.count('```') % 2 == 1:
        return text + '\n```'
    if re.search(r'(?:[.!?»"\')\]]|```)\s*$', text):
        return text
    last_fence = text.rfind('```')
    if last_fence != -1 and text.count('```') >= 2:
        tail = text[last_fence + 3:]
        punct = max(tail.rfind('.'), tail.rfind('!'), tail.rfind('?'))
        if punct != -1:
            return text[:last_fence + 3 + punct + 1].strip()
        return text[:last_fence + 3].strip()
    punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if punct >= 12:
        return text[:punct + 1].strip()
    return text


def get_gguf_model():
    global _gguf_model, _gguf_path
    with _gguf_lock:
        if _gguf_model is None:
            chosen = next((p for p in GGUF_CANDIDATES if p.is_file()), None)
            if chosen is None:
                raise FileNotFoundError('Nessun modello GGUF locale trovato in models/')
            from llama_cpp import Llama
            phys_threads = min(10, max(4, (os.cpu_count() or 8) // 3))
            batch_threads = min(16, max(phys_threads, (os.cpu_count() or 8) // 2))
            _gguf_model = Llama(
                model_path=str(chosen),
                n_ctx=2048,
                n_gpu_layers=-1,
                n_threads=phys_threads,
                n_threads_batch=batch_threads,
                n_batch=512,
                verbose=False,
            )
            _gguf_path = chosen
        return _gguf_model, _gguf_path


def _generate_gguf(turns, max_tokens=200, system=None):
    from lumina_answer_checks import enforce_format
    last_prompt = turns[-1]['content']
    low = last_prompt.lower()
    needs_long_budget = bool(re.search(
        r'\b(?:codice|python|javascript|typescript|html|css|sql|bash|powershell|funzion[ei]|script|class[ei]|programma|algoritm[oi]|approfondit[oa]|dettagliat[oa]|passo\s+(?:dopo\s+|per\s+)?passo|elenco\s+completo|tabella|racconto\s+lungo)\b',
        low,
    ))
    cap = 280 if needs_long_budget else 115
    max_new = max(1, min(cap, int(max_tokens or cap)))
    with _gguf_lock:
        llm, path = get_gguf_model()
        trimmed = list(turns[-8:])
        while trimmed and trimmed[0]['role'] == 'assistant':
            trimmed.pop(0)
        chat_messages = [{'role': 'system', 'content': system or SYSTEM}] + trimmed
        out = llm.create_chat_completion(
            messages=chat_messages,
            max_tokens=max_new,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
        )
    choice = out['choices'][0]
    raw_answer = choice['message']['content'] or ''
    finish_reason = choice.get('finish_reason')
    answer = _trim_unfinished_sentence(_sanitize_public_identity(raw_answer), finish_reason)
    answer = enforce_format(answer, last_prompt)
    usage = out.get('usage') or {}
    return answer, {
        'engine': 'Lumina Assistita Locale (GGUF)',
        'provider': 'llama.cpp-local',
        'model_id': path.parent.name,
        'local': True,
        'assisted': True,
        'completion_tokens': usage.get('completion_tokens', 0),
        'finished': finish_reason != 'length',
    }


def get_model():
    global _model,_tokenizer
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
            device='cuda' if torch.cuda.is_available() else 'cpu'
            dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
            tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,local_files_only=True)
            model=AutoModelForCausalLM.from_pretrained(MODEL_ID,local_files_only=True,dtype=dtype).to(device).eval()
            _tokenizer,_model=tokenizer,model
        return _model,_tokenizer


def get_next_local_model():
    global _next_model,_next_tokenizer
    with _lock:
        if _next_model is None:
            import torch
            from tokenizers import ByteLevelBPETokenizer
            from modeling_lumina_next import LuminaNextForCausalLM
            from lumina_next_runtime import active_model_dir
            device='cuda' if torch.cuda.is_available() else 'cpu'
            dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
            folder=active_model_dir()
            _next_tokenizer=ByteLevelBPETokenizer(str(folder/'vocab.json'),str(folder/'merges.txt'))
            _next_model=LuminaNextForCausalLM.from_pretrained(folder).to(device=device,dtype=dtype).eval()
        return _next_model,_next_tokenizer


def _generate_lumina_next(turns,max_tokens=200):
    import torch
    from lumina_next_runtime import active_model_dir,prompt_ids,stop_sequences,response_text
    from lumina_answer_checks import enforce_format
    with _lock:
        model,tok=get_next_local_model()
        dev=next(model.parameters()).device
        max_new=max(1,min(256,int(max_tokens or 200)))
        budget=min(2048,model.config.get('max_position_embeddings',2048)-max_new)
        ids=prompt_ids(tok,turns,budget=max(128,budget))
        inputs=torch.tensor([ids],device=dev,dtype=torch.long)
        with torch.inference_mode():
            out=model.generate(
                inputs,
                max_new_tokens=max_new,
                temperature=0.0,
                eos_token_id=1,
                stop_token_sequences=stop_sequences(tok),
            )
        gen_ids=out[0][len(ids):].tolist()
        answer=response_text(tok,gen_ids)
        answer=enforce_format(answer,turns[-1]['content'])
        return answer,{
            'engine':'Lumina Next PyTorch',
            'provider':'PyTorch-CUDA' if dev.type=='cuda' else 'PyTorch-CPU',
            'model_id':active_model_dir().name,
            'local':True,
            'assisted':True,
            'completion_tokens':len(gen_ids),
            'finished':True,
        }


def runtime_status():
    """Provider attivo per l'inferenza: PyTorch-CUDA, DirectML o CPU."""
    mode=os.getenv('LUMINA_RUNTIME','auto').lower()
    if mode!='directml':
        try:
            import torch
            if torch.cuda.is_available():
                return 'PyTorch-CUDA'
        except Exception:
            pass
    if mode in ('auto','directml'):
        try:
            from lumina_onnx_runtime import gpu_provider_available
            if gpu_provider_available(): return 'DmlExecutionProvider'
        except Exception: pass
    try:
        import torch
        return 'PyTorch-CUDA' if torch.cuda.is_available() else 'PyTorch-CPU'
    except Exception:
        return 'CPU'


def generate(messages,max_tokens=200,*,use_calculator=True,system=None):
    turns=[{'role':m['role'],'content':str(m.get('content',''))} for m in messages
           if m.get('role') in ('user','assistant')]
    if not turns or turns[-1]['role']!='user':raise ValueError('Serve un messaggio utente finale')
    if use_calculator:
        from lumina_answer_checks import local_answer
        result,tool=local_answer(turns)
        if result is not None:
            return result,{'engine':tool,'model_id':None,'local':True,'assisted':True}
        answer=calculate(turns[-1]['content'])
        if answer is not None:
            return answer,{'engine':'calcolatrice locale','model_id':None,'local':True,'assisted':True}
    global _onnx_runtime
    import torch
    mode=os.getenv('LUMINA_RUNTIME','auto').lower()
    if mode != 'directml':
        try:
            answer, meta = _generate_gguf(turns, max_tokens=max_tokens, system=system)
            if answer:
                return answer, meta
        except Exception:
            pass
    if mode == 'directml':
        try:
            if _onnx_runtime is None:
                from lumina_onnx_runtime import LuminaOnnxRuntime
                _onnx_runtime=LuminaOnnxRuntime()
            prompt=''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in turns)
            prompt += '<|im_start|>assistant\n'
            answer=_onnx_runtime.generate(prompt,max_new_tokens=max(1,min(256,max_tokens)))
            from lumina_answer_checks import enforce_format
            answer=enforce_format(answer,turns[-1]['content'])
            if answer:
                return answer,{'engine':'Lumina ONNX DirectML','provider':_onnx_runtime.provider_name,'model_id':'lumina_next_135m.onnx','local':True,'assisted':True}
        except Exception:
            pass
    with _lock:
        try:
            model,tok=get_model()
        except Exception:
            return _generate_lumina_next(turns,max_tokens=max_tokens)
        context=[{'role':'system','content':system or SYSTEM}]+turns
        while True:
            text=tok.apply_chat_template(context,tokenize=False,add_generation_prompt=True)
            inputs=tok(text,return_tensors='pt').to(next(model.parameters()).device)
            if inputs.input_ids.shape[1]<=4096:break
            if len(context)<=2:raise ValueError('Messaggio troppo lungo per il contesto locale')
            context.pop(1)
            while len(context)>2 and context[1]['role']=='assistant':context.pop(1)
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=max(1,min(512,max_tokens)),do_sample=False,pad_token_id=tok.eos_token_id)
        answer=_sanitize_public_identity(tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip())
        from lumina_answer_checks import enforce_format
        answer=enforce_format(answer,turns[-1]['content'])
        return answer,{'engine':'Modello locale','model_id':MODEL_ID,'local':True,'assisted':True,
                       'completion_tokens':out.shape[1]-inputs.input_ids.shape[1],
                       'finished':int(out[0,-1])==tok.eos_token_id}
