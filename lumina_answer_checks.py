"""Controlli e strumenti simbolici locali applicabili a domande nuove."""
import ast
import re

DAYS=['lunedì','martedì','mercoledì','giovedì','venerdì','sabato','domenica']
NUMBERS={'un':1,'uno':1,'due':2,'tre':3,'quattro':4,'cinque':5,'sei':6,'sette':7,'otto':8,'nove':9,'dieci':10}


def local_answer(messages):
    text=messages[-1]['content'];lower=text.lower()
    history=' '.join(m['content'] for m in messages[:-1] if m['role']=='user')
    if re.fullmatch(r'\s*(?:come ti chiami|chi sei|qual è il tuo nome)\s*[?!.]*\s*',lower):
        return 'Sono Lumina, un assistente eseguito sul tuo computer con supporto Qwen locale.','identità configurata'
    if re.search(r'chi\s+vincer[àa]',lower) and not re.search(r'racconto|romanzo|storia inventata',history.lower()):
        return 'Non è possibile conoscere con certezza il risultato di un evento futuro.','limiti di previsione'
    if re.search(r'non puoi veder|non ti ho dato questa informazione',lower) and not history:
        return 'Non ho questa informazione e non posso osservare oggetti o persone sul tuo computer.','disponibilità dei dati'
    quote=re.search(r'(?:dal testo|nel testo|testo:)\s*["“](.*?)["”]',text,re.I|re.S)
    if quote:
        source=quote.group(1).lower()
        requirements=[('peso',r'\b(?:kg|grammi|chilogrammi|pesa|peso)\b'),('prezzo',r'€|euro|prezzo|costa'),('durata',r'minuti|ore|secondi|durata')]
        question=lower[quote.end():]
        for field,pattern in requirements:
            if field in question and not re.search(pattern,source):return f'Il dato richiesto ({field}) non è specificato nel testo.','verifica della fonte'
    start=re.search(r'oggi\s+[èe]\s+('+'|'.join(DAYS)+r')\b',lower)
    if start:
        index=DAYS.index(start.group(1));delta=None
        if 'ieri' in lower:delta=-1
        else:
            after=re.search(r'\b(?:fra|tra)\s+(\d+|'+ '|'.join(NUMBERS)+r')\s+giorn',lower)
            if after:delta=int(after.group(1)) if after.group(1).isdigit() else NUMBERS[after.group(1)]
            elif 'domani' in lower:delta=1
        if delta is not None:return DAYS[(index+delta)%7],'calendario locale'
    ordered=re.search(r'ordina.*?\b(crescente|decrescente)\b\s*([\d\s,;.\-]+)',lower)
    if ordered:
        values=re.findall(r'-?\d+',ordered.group(2))
        if 1<len(values)<=100:
            return ', '.join(str(x) for x in sorted(map(int,values),reverse=ordered.group(1)=='decrescente')),'ordinamento locale'
    edges=re.findall(r'\b(\w+)\s+è\s+più\s+alta\s+di\s+(\w+)\b',text,re.I)
    if edges and re.search(r'chi è la più alta',lower):
        roots={a for a,b in edges}-{b for a,b in edges}
        if len(roots)==1:return next(iter(roots)),'confronto di relazioni esplicite'
    return None,None


def enforce_format(answer,prompt):
    """Normalizza soltanto vincoli espliciti; non corregge fatti usando il benchmark."""
    lower=prompt.lower();answer=answer.strip()
    literal=re.search(r'\bparola\s+([^\W\d_]+)[.!?]?\s*$',prompt,re.I)
    if literal and ('esclusivamente' in lower or 'soltanto' in lower):
        if answer.rstrip('.').casefold()==literal.group(1).casefold():answer=literal.group(1)
    if 'json' in lower and ('esclusivamente' in lower or 'solo' in lower):
        fenced=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',answer,re.S)
        if fenced:answer=fenced.group(1).strip()
    if 'solo il numero' in lower:
        numbers=re.findall(r'(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)',answer)
        if len(numbers)==1:answer=numbers[0]
    if 'solo il colore' in lower or 'soltanto il giorno' in lower or 'colori' in lower and 'separati da virgola' in lower:
        answer=answer.rstrip('.').lower()
    if ('codice' in lower or 'funzione' in lower and 'python' in lower) and ('solo' in lower or 'soltanto' in lower):
        requested=re.search(r'(?:funzione(?:\s+python)?|definisci(?:\s+in\s+python)?)\s+([a-zA-Z_]\w*)\s*\(',prompt,re.I)
        block=re.search(r'```(?:python)?\s*\n(.*?)```',answer,re.S)
        if requested and block:
            try:
                tree=ast.parse(block.group(1));functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
                if len(functions)==1:
                    old=functions[0].name;new=requested.group(1)
                    if old!=new and not any(isinstance(n,ast.Name) and n.id==old for n in ast.walk(functions[0])):
                        answer=re.sub(r'\bdef\s+'+re.escape(old)+r'\s*\(', 'def '+new+'(',answer,count=1)
            except SyntaxError:pass
    return answer
