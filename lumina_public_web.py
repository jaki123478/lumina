"""Ricerca e lettura di pagine pubbliche; modelli e TTS restano locali."""
from html.parser import HTMLParser
from urllib.parse import urlparse,quote
from datetime import datetime,timezone
import socket,ipaddress,re,xml.etree.ElementTree as ET
import requests
from lumina_network_policy import get_network_policy

def validate_url(url):
    p=urlparse(url)
    if p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in (None,443):
        raise ValueError('Sono supportate pagine pubbliche HTTPS senza credenziali')
    for item in socket.getaddrinfo(p.hostname,443,type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(item[4][0]).is_global:raise ValueError('Indirizzo privato o locale non ammesso per la lettura Internet')
    return url

def fetch(url):
    if not get_network_policy().is_internet_allowed():raise PermissionError('Internet disattivato')
    s=requests.Session();s.trust_env=False
    for _ in range(4):
        validate_url(url)
        r=s.get(url,timeout=(8,20),allow_redirects=False,stream=True,headers={'User-Agent':'LuminaLocal/1.0 public reader'})
        if r.is_redirect:
            from urllib.parse import urljoin
            url=urljoin(url,r.headers['Location']);r.close();continue
        r.raise_for_status();parts=[];size=0
        for chunk in r.iter_content(32768):
            size+=len(chunk)
            if size>8_000_000:
                if 'html' in r.headers.get('Content-Type',''):
                    break  # Estratto HTML limitato, non scartare tutte le pagine grandi.
                raise ValueError('Documento oltre il limite di 8 MB')
            parts.append(chunk)
        return b''.join(parts),r.headers.get('Content-Type',''),url
    raise ValueError('Troppi reindirizzamenti')

class Extractor(HTMLParser):
    def __init__(self):super().__init__();self.skip=0;self.title=False;self.title_parts=[];self.parts=[];self.paragraphs=[];self.paragraph=None
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','nav','noscript'):self.skip+=1
        if tag=='title':self.title=True
        if tag=='p':self.paragraph=[]
    def handle_endtag(self,tag):
        if tag in ('script','style','nav','noscript') and self.skip:self.skip-=1
        if tag=='title':self.title=False
        if tag=='p' and self.paragraph is not None:
            self.paragraphs.append(' '.join(' '.join(self.paragraph).split()));self.paragraph=None
    def handle_data(self,data):
        if not self.skip:
            self.parts.append(data)
            if self.title:self.title_parts.append(data)
            if self.paragraph is not None:self.paragraph.append(data)

def search(query):
    # Solo le parole dell'ultima ricerca vanno al motore, mai la cronologia o l'audio.
    query=re.sub(r"\b(?:dimmi|spiegami|che\s+cos(?:['’`]è|\s+è)|cos(?:['’`]è|\s+è)|cosa\s+è)\b", " ", query, flags=re.I)
    query=re.sub(r"\s+", " ", query).strip()
    query=re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b','[email rimossa]',query,flags=re.I)
    query=re.sub(r'\b\d{8,}\b','[numero rimosso]',query).strip()[:180]
    if not query:raise ValueError('Inserisci le parole da cercare')
    raw,_,_=fetch('https://www.bing.com/search?format=rss&q='+quote(query))
    root=ET.fromstring(raw);results=[];domains=set();candidates=[]
    sites=re.findall(r'\bsite:([a-z0-9.-]+)',query.lower())
    for node in root.findall('.//item'):
        link=node.findtext('link','');host=urlparse(link).hostname
        if not host or urlparse(link).scheme not in ('http','https'):continue
        if urlparse(link).scheme=='http':link=urlparse(link)._replace(scheme='https').geturl()
        if sites and not any(host==site or host.endswith('.'+site) for site in sites):continue
        if len(candidates)<10:candidates.append({'url':link,'title':''})
        if host in domains:continue
        domains.add(host)
        results.append({'title':' '.join(node.findtext('title','').split()[:8]),'excerpt':' '.join(re.sub('<[^>]+>',' ',node.findtext('description','')).split()[:17]),'url':link})
        if len(results)>4:results=results[:4]
    if not results:raise ValueError('Il motore di ricerca non ha restituito risultati leggibili')
    lines=['Risultati di ricerca, non fatti già verificati. Controlla le pagine indicate per data e dettagli.']
    for row in results:lines.append(f"{row['title']}. {row['excerpt']}\nFonte: {row['url']}")
    return '\n\n'.join(lines),{'engine':'ricerca web pubblica','external_retrieval':True,'query_sent':True,'query':query,'sources':results,'candidates':candidates,'audio_processing':'local','retrieved_at':datetime.now(timezone.utc).isoformat()}

def read_page(url):
    raw,kind,final=fetch(url)
    if 'html' not in kind and 'text/plain' not in kind:raise ValueError('Formato non supportato: serve una pagina testuale')
    parser=Extractor();parser.feed(raw.decode('utf-8',errors='replace'))
    title=' '.join(' '.join(parser.title_parts).split()[:8]);excerpt=' '.join(' '.join(parser.parts).split()[:17])
    return f'Pagina consultata: {title}. Breve estratto: {excerpt}\nFonte: {final}',{'engine':'lettura pagina pubblica','external_retrieval':True,'audio_processing':'local','sources':[{'url':final,'title':title}],'retrieved_at':datetime.now(timezone.utc).isoformat()}

def answer(prompt):
    urls=re.findall(r'https?://[^\s<>"\']+',prompt)
    try:
        if urls:return read_page(urls[0].rstrip('.,;)'))
        query=re.sub(r'^\s*(?:cerca(?:\s+(?:su internet|sul web|in rete))?|ricerca)\s*[:,-]?\s*','',prompt,flags=re.I)
        return search(query)
    except Exception as e:
        return 'Non sono riuscita a verificare questa richiesta sul web: '+str(e),{'engine':'web pubblico','external_retrieval':True,'error':str(e),'sources':[],'audio_processing':'local'}
