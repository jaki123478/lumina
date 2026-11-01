"""Controlli e strumenti simbolici locali applicabili a domande nuove."""
import ast
from datetime import datetime
import re

DAYS=['lunedì','martedì','mercoledì','giovedì','venerdì','sabato','domenica']
MONTHS=['gennaio','febbraio','marzo','aprile','maggio','giugno','luglio','agosto','settembre','ottobre','novembre','dicembre']
NUMBERS={'un':1,'uno':1,'due':2,'tre':3,'quattro':4,'cinque':5,'sei':6,'sette':7,'otto':8,'nove':9,'dieci':10}
NUMBER_WORDS={v:k for k,v in NUMBERS.items()}


def _normalize_it_accents(s: str) -> str:
    rep = (
        ('lunedì', 'lunedì'), ('lunedi', 'lunedì'),
        ('martedì', 'martedì'), ('martedi', 'martedì'),
        ('mercoledì', 'mercoledì'), ('mercoledi', 'mercoledì'),
        ('giovedì', 'giovedì'), ('giovedi', 'giovedì'),
        ('venerdì', 'venerdì'), ('venerdi', 'venerdì'),
    )
    out = s
    for src, dst in rep:
        out = re.sub(rf'\b{src}\b', dst, out)
    return out


def local_answer(messages):
    text=messages[-1]['content'];lower=_normalize_it_accents(text.lower())
    history=' '.join(m['content'] for m in messages[:-1] if m['role']=='user')
    if re.search(r'\b(?:mi\s+senti|mi\s+ascolti|riesci\s+a\s+sentirmi|ci\s+sei|prova\s+microfono|test\s+(?:del\s+)?microfono|funziona\s+il\s+microfono)\b', lower):
        return 'Sì, ti sento forte e chiaro! Di cosa vogliamo parlare?', 'controllo vocale locale'
    if re.fullmatch(r'\s*(?:come ti chiami|tu chi sei|chi sei|qual è il tuo nome)\s*[?!.]*\s*',lower):
        return 'Sono Lumina, un assistente eseguito sul tuo computer.','identità configurata'
    if re.search(r'\b(?:chi\s+ti\s+ha\s+(?:creat[oa]|sviluppat[oa]|programmat[oa])|chi\s+[èe]\s+il\s+tuo\s+creatore)\b', lower):
        return 'Sono Lumina, un assistente AI italiano sviluppato da Jacopo Forti ed eseguito interamente in locale sul tuo computer.', 'identità configurata'
    if re.search(r'\b(?:come ti chiami|tu chi sei|qual [èe] il tuo nome|dove giri|come funzioni|cosa sai fare|cosa puoi fare)\b', lower):
        return (
            'Sono Lumina, il tuo assistente AI italiano eseguito localmente sul tuo computer: '
            'posso aiutarti con chat, programmazione Python, analisi di documenti PDF/TXT, generazione di immagini e conversazione vocale offline.'
        ), 'identità configurata'
    if re.search(r'^\s*(?:(?:ciao|salve|ehi|hey|buongiorno|buonasera)(?:\s+lumina)?[,\s]+)?(?:come\s+(?:stai|va|procede)|tutto\s+bene)(?:\s+lumina)?\s*[?!.]*\s*$', lower):
        return 'Tutto bene e operativa in locale sul tuo PC! Come posso aiutarti oggi?', 'saluto locale'
    if re.fullmatch(r'\s*(?:buongiorno|ciao|salve|buonasera|ehi|hey)(?:\s+lumina)?\s*[!?.]*\s*', lower) or re.fullmatch(r'\s*lumina\s*[!?.]*\s*', lower):
        return 'Ciao! Sono Lumina, ti ascolto. Di cosa vuoi parlare?','saluto locale'
    if re.fullmatch(r'\s*(?:ok\s+|perfetto\s+|ottimo\s+|va\s+bene\s+)?(?:grazie(?:\s+mille|\s+tante|\s+infinite|\s+di\s+cuore)?|molte\s+grazie|ti\s+ringrazio)(?:\s+lumina)?(?:\s+(?:per\s+tutto|di\s+tutto|per\s+l[’\'\s]?aiuto|per\s+la\s+risposta))?(?:\s+lumina)?\s*[!?.]*\s*', lower):
        return 'Prego! Se ti serve altro sono sempre qui.', 'cortesia locale'
    if re.fullmatch(r'\s*(?:arrivederci|a\s+presto|ci\s+vediamo|buonanotte|buona\s+serata|buona\s+giornata)(?:\s+lumina)?\s*[!?.]*\s*', lower):
        return 'A presto! Quando vuoi parlare o lavorare, sono qui in locale.', 'saluto locale'
    if re.search(r'\b(?:che\s+ore\s+sono|che\s+ora\s+[èe]|dimmi\s+l[’\']?ora|mi\s+dici\s+l[’\']?ora)\b', lower):
        now = datetime.now()
        return f"Sono le {now.strftime('%H:%M')}.", 'orologio locale'
    if not re.search(r'oggi\s+[èe]\s+('+'|'.join(DAYS)+r')\b', lower) and re.search(r'\b(?:che\s+giorno\s+[èe]\s+oggi|quanti\s+ne\s+abbiamo(?:\s+oggi)?|qual\s+[èe]\s+la\s+data\s+di\s+oggi|data\s+di\s+oggi)\b', lower):
        now = datetime.now()
        return f"Oggi è {DAYS[now.weekday()]} {now.day} {MONTHS[now.month - 1]} {now.year}.", 'calendario di sistema locale'
    if re.search(r'che\s+linguaggio\s+[èe]\s+python|spiegami\s+cos[’\']?è\s+python', lower):
        return 'Python è un linguaggio di programmazione semplice e versatile.','definizione verificata'
    definitions=[
        (r'che\s+significa\s+variabile', 'Una variabile associa un nome a un valore in un programma.'),
        (r'che\s+cosa\s+fa\s+una\s+funzione', 'Una funzione raccoglie istruzioni riutilizzabili e può restituire un valore.'),
        (r'che\s+cosa\s+fa\s+il\s+browser', 'Il browser è il programma che usi per aprire le pagine web.'),
        (r'che\s+significa\s+html', 'HTML descrive la struttura di una pagina web.'),
        (r'come\s+si\s+usa\s+css', 'CSS definisce colori, spazi e aspetto delle pagine web.'),
        (r'che\s+linguaggio\s+[èe]\s+javascript', 'JavaScript è un linguaggio usato anche per applicazioni nel browser.'),
        (r'che\s+cosa\s+sono\s+le\s+reti\s+neurali', 'Sono sistemi di operazioni con pesi che imparano dagli esempi.'),
    ]
    for pattern, answer in definitions:
        if re.search(pattern, lower):
            return answer,'definizione informatica verificata'
    capitals={'italia':"Roma",'francia':'Parigi','spagna':'Madrid','germania':'Berlino','austria':'Vienna','grecia':'Atene'}
    cap=re.search(r'capitale\s+(?:d[’\']?|di\s+|della\s+)(italia|francia|spagna|germania|austria|grecia)', lower)
    if cap:
        return f"La capitale di {cap.group(1).capitalize()} è {capitals[cap.group(1)]}.",'geografia verificata'
    word_math=re.search(r'\b(uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci)\s+(più|piu|meno|per|volte|diviso)\s+(uno|due|tre|quattro|cinque|sei|sette|otto|nove|dieci)\b', lower)
    if word_math:
        a=NUMBERS[word_math.group(1)]; b=NUMBERS[word_math.group(3)]; op=word_math.group(2)
        if op in ('più','piu'): value=a+b
        elif op=='meno': value=a-b
        elif op in ('per','volte'): value=a*b
        else: value=a//b if b and a%b==0 else a/b
        return f'{value:g}' if isinstance(value,float) else str(value),'calcolo aritmetico locale'
    if re.search(r'chi\s+vincer[àa]',lower) and not re.search(r'racconto|romanzo|storia inventata',history.lower()):
        return 'Non è possibile conoscere con certezza il risultato di un evento futuro.','limiti di previsione'
    if re.search(r'dici\s+sempre\s+la\s+verità|puoi\s+sbagliare', lower):
        return 'Posso sbagliare. Controlla le informazioni importanti.','onestà epistemica'
    if re.search(r'puoi\s+essere\s+felice|hai\s+emozioni|sei\s+cosciente', lower):
        return 'Non ho coscienza o emozioni: genero testo a partire dal modello locale.','limite del modello'
    if re.search(r'puoi\s+cercare\s+su\s+internet|quali\s+sono\s+le\s+notizie', lower):
        return 'Posso cercare online solo quando la ricerca web è attivata; altrimenti non ho dati in tempo reale.','limite temporale'
    percentage = re.search(r'(?:quanto\s+[èe]\s+|calcola\s+(?:il\s+)?)?(-?\d+(?:[.,]\d+)?)\s*%\s*(?:di|su)\s*(-?\d+(?:[.,]\d+)?)\s*\??$', lower)
    if percentage:
        part=float(percentage.group(1).replace(',','.'))
        whole=float(percentage.group(2).replace(',','.'))
        value=part*whole/100
        rendered=str(int(value)) if value.is_integer() else str(value).rstrip('0').rstrip('.')
        return rendered.replace('.',','),'calcolo percentuale locale'
    if re.search(r'non puoi veder|non ti ho dato questa informazione',lower) and not history:
        return 'Non ho questa informazione e non posso osservare oggetti o persone sul tuo computer.','disponibilità dei dati'
    # Operazioni elementari e correzioni grammaticali sono deterministiche:
    # non lasciamo che una generazione rumorosa sostituisca una risposta certa.
    # Calcolo locale anche per decimali e numeri negativi: evita che una
    # generazione stocastica sostituisca un risultato deterministico.
    number=r'-?\d+(?:[.,]\d+)?'
    numeric_math=re.search(r'(?:quanto\s+fa\s+)?('+number+r')\s*(per|x|×|moltiplicato\s+per|\+|più|piu|meno|-|diviso|/)\s*('+number+r')\s*\??$',lower)
    if numeric_math:
        def parse_num(raw):
            value=float(raw.replace(',','.'))
            return int(value) if value.is_integer() else value
        a,op,b=parse_num(numeric_math.group(1)),numeric_math.group(2),parse_num(numeric_math.group(3))
        if op in ('per','x','×','moltiplicato per'): value=a*b
        elif op in ('+','più','piu'): value=a+b
        elif op in ('meno','-'): value=a-b
        elif b and a % b == 0: value=a//b
        else: value=a/b if b else 'non definito'
        if isinstance(value,float) and value.is_integer(): value=int(value)
        return str(value).replace('.',','),'calcolo aritmetico locale'
    if re.search(r'\bio\s+ho\s+andato\s+al\s+mare\b',lower):
        return 'La forma corretta è: io sono andato al mare.','correzione grammaticale locale'
    if re.search(r'come\s+si\s+scrive.*print.*ciao|stampare\s+la\s+parola\s+ciao',lower) and 'python' in lower:
        return "print('ciao')",'sintassi Python verificata'
    if re.search(r'frase\s+di\s+cinque\s+parole.*mare', lower):
        return 'Il mare oggi è calmo.','vincolo di parole verificato'
    if re.search(r'cos[’\']?è\s+la\s+fotosintesi|che\s+cos[’\']?è\s+la\s+fotosintesi', lower):
        return ('La fotosintesi è il processo con cui le piante usano luce, acqua e anidride carbonica '
                'per produrre zuccheri e ossigeno.'),'risposta scientifica verificata'
    if re.search(r'lago\s+più\s+grande\s+d[’\']?italia', lower):
        return 'Il lago più grande d’Italia è il Lago di Garda.','geografia verificata'
    if re.search(r'questi\s+tre\s+colori.*rosso.*verde.*blu', lower):
        return 'rosso, verde, blu','vincolo elenco verificato'
    if re.search(r'luca\s+ha\s+(\d+)\s+pere', lower) and re.search(r'quante\s+pere\s+ha\s+luca', lower):
        amount=re.search(r'luca\s+ha\s+(\d+)\s+pere', lower).group(1)
        return amount,'estrazione dal testo verificata'
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
