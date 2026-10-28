"""Supporto Qwen e calcolo locale. Il backend è dichiarato nelle risposte API."""
import ast
from fractions import Fraction
import os
import re
import threading

for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key]='1'
MODEL_ID='Qwen/Qwen2.5-Coder-1.5B-Instruct'
_lock=threading.RLock()
_model=None
_tokenizer=None
SYSTEM=("Sei Lumina, assistente in italiano eseguito sul computer dell'utente con supporto Qwen locale. "
        "Rispondi direttamente alla domanda, senza preamboli. Rispetta esattamente nomi, numeri, formato e numero di elementi richiesti. "
        "Usa tutti i messaggi precedenti per recuperare i dati della conversazione. Non inventare nomi, dati privati, eventi futuri o fonti. "
        "Se un dato manca, dichiara precisamente quale informazione manca. Per codice restituisci la funzione completa con il nome richiesto; "
        "non aggiungere stampe, test o spiegazioni quando è richiesto solo codice. Non fingere di avere accesso a Internet o sensori.")


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


def generate(messages,max_tokens=200,*,use_calculator=True,system=None):
    turns=[{'role':m['role'],'content':str(m.get('content',''))} for m in messages
           if m.get('role') in ('user','assistant')]
    if not turns or turns[-1]['role']!='user':raise ValueError('Serve un messaggio utente finale')
    if use_calculator:
        answer=calculate(turns[-1]['content'])
        if answer is not None:
            return answer,{'engine':'calcolatrice locale','model_id':None,'local':True,'assisted':True}
    with _lock:
        import torch
        model,tok=get_model()
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
        answer=tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
        return answer,{'engine':'Qwen locale','model_id':MODEL_ID,'local':True,'assisted':True,
                       'completion_tokens':out.shape[1]-inputs.input_ids.shape[1],
                       'finished':int(out[0,-1])==tok.eos_token_id}
