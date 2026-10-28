"""Aggiornamenti da feed ufficiali fissi. Nessun audio o dialogo inviato ai siti."""
from pathlib import Path
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json,re,os,xml.etree.ElementTree as ET
from urllib.parse import urlparse
import requests
ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'config/lumina_web_sources.json'
SOURCES={
 'ambiente':('ISPRA','https://www.isprambiente.gov.it/it/news/notizie/RSS','isprambiente.gov.it'),
 'protezione civile':('Protezione Civile','https://api.protezionecivile.it/default/dpcPortalGenerateRss?categoria=notizia','protezionecivile.gov.it'),
 'spazio':('ESA Italia','https://www.esa.int/rssfeed/Italy','esa.int')}

def enabled():
    if os.environ.get('LUMINA_ALLOW_INTERNET','').lower() in ('0','false','no'):return False
    try:return json.loads(CONFIG.read_text(encoding='utf-8')).get('enabled',False) is True
    except (OSError,ValueError):return False

def set_enabled(value):
    CONFIG.write_text(json.dumps({'enabled':bool(value),'mode':'explicit_official_feeds','audio_processing':'local','automatic_training':False},indent=2),encoding='utf-8')

def text(value):
    value=unescape(re.sub('<[^>]+>',' ',value or ''))
    value=value.replace('\x92','’')
    return ' '.join(value.split())

def parse_feed(raw,name,domain):
    root=ET.fromstring(raw);rows=[]
    nodes=[node for node in root.iter() if node.tag.split('}')[-1] in ('item','entry')]
    def field(node,key):
        return next((n.text or '' for n in node if n.tag.split('}')[-1]==key),'')
    for node in nodes:
        title=text(field(node,'title'));link=field(node,'link')
        if not link:
            link=next((n.get('href','') for n in node if n.tag.split('}')[-1]=='link' and n.get('rel','alternate')=='alternate'),'')
        host=urlparse(link).hostname or ''
        if urlparse(link).scheme!='https' or not (host==domain or host.endswith('.'+domain)):continue
        date=field(node,'pubDate') or field(node,'published') or field(node,'updated') or field(node,'date')
        try:
            dt=parsedate_to_datetime(date) if ',' in date else datetime.fromisoformat(date.replace('Z','+00:00'))
            if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        except (ValueError,TypeError):continue
        if dt>datetime.now(timezone.utc):continue
        if title:rows.append({'source':name,'title':' '.join(title.split()[:20]),'url':link,'published':dt.isoformat()})
    return sorted(rows,key=lambda r:r['published'],reverse=True)

def latest(topic=''):
    if not enabled():raise PermissionError('Accesso ai feed disattivato. Scrivi: autorizzo internet.')
    selected=[key for key in SOURCES if key in topic.lower()]
    if not selected:selected=list(SOURCES)
    result=[];errors=[]
    session=requests.Session();session.trust_env=False
    for key in selected:
        name,url,domain=SOURCES[key]
        try:
            response=session.get(url,timeout=(8,20),allow_redirects=False,stream=True,headers={'User-Agent':'LuminaLocal/1.0 feed reader'})
            if response.status_code!=200:raise ValueError('HTTP '+str(response.status_code))
            chunks=[];size=0
            for chunk in response.iter_content(32768):
                size+=len(chunk)
                if size>2_000_000:raise ValueError('Feed troppo grande')
                chunks.append(chunk)
            rows=parse_feed(b''.join(chunks),name,domain)
            if not rows:raise ValueError('Nessuna notizia con data e collegamento validi')
            preferred=[row for row in rows if '/it/' in row['url']] if key=='protezione civile' else rows
            result.append((preferred or rows)[0])
        except Exception as e:errors.append({'source':name,'error':str(e)})
    archive={'retrieved_at':datetime.now(timezone.utc).isoformat(),'items':result,'errors':errors,'review_required_before_training':True}
    folder=ROOT/'data/web_sources';folder.mkdir(parents=True,exist_ok=True)
    (folder/'latest.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
    return archive

def answer(topic):
    data=latest(topic)
    if not data['items']:return 'Non riesco a verificare aggiornamenti dalle fonti ufficiali in questo momento.',{'engine':'feed ufficiali','external_retrieval':True,'sources':[],'errors':data['errors']}
    lines=['Consulto soltanto i feed ufficiali ISPRA, ESA e Protezione Civile. Questi sono gli ultimi titoli disponibili; le date sono quelle di pubblicazione.']
    for row in data['items']:lines.append(f"{row['source']} — {row['published'][:10]}: {row['title']}\nFonte: {row['url']}")
    return '\n\n'.join(lines),{'engine':'feed ufficiali, lettura locale','external_retrieval':True,'audio_processing':'local','query_sent':False,'sources':data['items'],'retrieved_at':data['retrieved_at'],'errors':data['errors']}
