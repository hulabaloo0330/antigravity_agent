#!/usr/bin/env python3
"""Bounded C. elegans behavior/neural discovery and download agent.

Version 2 adds reviewed NumPy/HDF5 adapters. Unknown numerical formats remain
unparsed; verified downloads are NEVER automatically called training-ready.
"""
import argparse, concurrent.futures, csv, datetime as dt, fcntl, gzip, hashlib
import html, ipaddress, json, os, pathlib, re, shutil, signal, socket, sqlite3
import threading, time, urllib.error, urllib.parse, urllib.request, uuid
import xml.etree.ElementTree as ET

ROOT=pathlib.Path(os.environ.get('WORM_ROOT','/opt/worm-collector'))
NOW=lambda: dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
VERSION='1.0.0'
PROGRAM_VERSION='2.0.1'
ALLOWED=('ncbi.nlm.nih.gov','ebi.ac.uk','europepmc.org','zenodo.org','datadryad.org',
         'github.com','api.github.com','raw.githubusercontent.com','objects.githubusercontent.com',
         'bitbucket.org','api.bitbucket.org','wormwideweb.org','figshare.com','osf.io',
         'doi.org','api.deepseek.com','amazonaws.com','cloudfront.net','storage.googleapis.com')
REPOSITORIES=('zenodo.org','datadryad.org','github.com','bitbucket.org','wormwideweb.org','figshare.com','osf.io')
SUFFIXES=('.csv','.tsv','.json','.h5','.hdf5','.nwb','.mat','.npz','.npy','.zip','.tar.gz','.tar.bz2','.bz2','.gz','.txt','.pkl')
STOP=threading.Event()

class BudgetError(Exception): pass
class ReviewError(Exception): pass

def safe_url(url):
    p=urllib.parse.urlsplit(url)
    host=(p.hostname or '').lower()
    if p.scheme not in ('http','https') or p.username or p.password or p.port not in (None,80,443):
        raise ReviewError('UNSAFE_URL')
    if not any(host==h or host.endswith('.'+h) for h in ALLOWED):
        raise ReviewError('UNSUPPORTED_HOST')
    addresses=socket.getaddrinfo(host,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
        raise ReviewError('NONPUBLIC_ADDRESS')
    return url

class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        safe_url(newurl)
        if req.has_header('Authorization') and urllib.parse.urlsplit(newurl).netloc!=urllib.parse.urlsplit(req.full_url).netloc:
            raise ReviewError('AUTH_REDIRECT_REFUSED')
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def xmltext(node):
    return ' '.join(' '.join(node.itertext()).split()) if node is not None else ''

def pubmed_article_ids(article):
    # ReferenceList also contains ArticleId nodes: never search all descendants.
    return {x.get('IdType'):xmltext(x) for x in article.findall('./PubmedData/ArticleIdList/ArticleId')}

class Agent:
    def __init__(self):
        self.root=ROOT;self.now=NOW;self.stopping=STOP.is_set
        self.policy=json.loads((ROOT/'config/policy.json').read_text())
        for name in ['state','papers','datasets','objects/sha256','incoming','review','logs','exports']:
            (ROOT/name).mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.netlock=threading.Lock();self.hostlocks={};self.hostnext={}
        self.db=sqlite3.connect(ROOT/'state/catalog.sqlite',check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON');self.db.execute('PRAGMA busy_timeout=5000')
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='papers'").fetchone():
            self.db.executescript((ROOT/'config/schema.sql').read_text())
        self.db.execute('CREATE TABLE IF NOT EXISTS runtime_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        if (ROOT/'config/neural_schema.sql').is_file():self.db.executescript((ROOT/'config/neural_schema.sql').read_text())
        self.db.commit()
        self.started=time.monotonic();self.deadline=self.started+1000
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirects())
        self.occupied=self.scan_size();self.reserved=0
        self.net_started=time.monotonic();self.net_bytes=0

    def scan_size(self):
        return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())

    def rows(self,sql,args=()):
        with self.lock:return [dict(x) for x in self.db.execute(sql,args).fetchall()]

    def run(self,sql,args=()):
        with self.lock:
            c=self.db.execute(sql,args);self.db.commit();return c.lastrowid

    def kv(self,key,default=None):
        row=self.rows('SELECT value FROM runtime_kv WHERE key=?',(key,))
        return json.loads(row[0]['value']) if row else default

    def setkv(self,key,value):
        self.run('INSERT INTO runtime_kv VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(value)))

    def model_budget_available(self):
        cap=self.policy.get('daily_llm_calls_max',300)
        return cap is None or self.kv('llm_calls:'+NOW()[:10],0)<cap

    def log(self,event,**fields):
        with self.lock:
            p=ROOT/'logs/events.jsonl'
            if p.exists() and p.stat().st_size>5_000_000:
                for i in (2,1):
                    old=ROOT/'logs'/f'events.{i}.jsonl'
                    if old.exists():os.replace(old,ROOT/'logs'/f'events.{i+1}.jsonl')
                os.replace(p,ROOT/'logs/events.1.jsonl')
            with p.open('a',encoding='utf-8') as f:f.write(json.dumps({'at':NOW(),'event':event,**fields},ensure_ascii=False)+'\n')

    def reserve(self,n):
        with self.lock:
            free=shutil.disk_usage(ROOT).free
            if self.occupied+self.reserved+n>18_000_000_000 or free-self.reserved-n<2_000_000_000:
                raise BudgetError('STORAGE_BUDGET')
            self.reserved+=n

    def release(self,n,written=0):
        with self.lock:self.reserved-=n;self.occupied+=written

    def charge(self,n):
        with self.lock:
            k='transfer:'+NOW()[:10];used=self.kv(k,0)
            cap=self.policy.get('daily_transfer_max_bytes')
            if cap is not None and used+n>cap:raise BudgetError('DAILY_TRANSFER_BUDGET')
            self.setkv(k,used+n)

    def pause_host(self,host,seconds):
        with self.netlock:self.hostnext[host]=max(self.hostnext.get(host,0),time.monotonic()+seconds)

    def open(self,url,data=None,headers=None):
        safe_url(url);host=urllib.parse.urlsplit(url).hostname
        with self.netlock:
            delay=max(0,self.hostnext.get(host,0)-time.monotonic())
            self.hostnext[host]=time.monotonic()+delay+1
        if delay>120:raise ReviewError('HOST_BACKOFF')
        if STOP.wait(delay):raise BudgetError('STOPPING')
        h={'User-Agent':'WormBehaviorCollector/1.0 (research; bounded single VPS)', 'Accept-Encoding':'identity',**(headers or {})}
        try:return self.opener.open(urllib.request.Request(url,data=data,headers=h),timeout=45)
        except urllib.error.HTTPError as e:
            if e.code==429:
                try:delay=max(60,min(86400,int(e.headers.get('Retry-After','300'))))
                except ValueError:delay=300
                self.pause_host(host,delay)
                self.setkv('host_backoff:'+host,time.time()+delay)
            raise

    def fetch(self,url,limit=2_000_000,data=None,headers=None):
        self.reserve(limit)
        try:
            with self.open(url,data,headers) as response:
                body=bytearray()
                while True:
                    chunk=response.read(min(65536,limit+1-len(body)))
                    if not chunk:break
                    self.charge(len(chunk));body.extend(chunk)
                    if len(body)>limit:raise ReviewError('METADATA_TOO_LARGE')
                return bytes(body)
        finally:self.release(limit)

    def fetch_json(self,url,**kwargs):return json.loads(self.fetch(url,**kwargs))

    def enqueue(self,action,target_type,target_id,payload=None,priority=50):
        revision='2.0.0' if action in ('screen','parse') else VERSION
        key=f'{revision}:{action}:{target_type}:{target_id}'
        self.run('INSERT OR IGNORE INTO jobs (dedupe_key,action,target_type,target_id,payload_json,priority,next_run_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)',
                 (key,action,target_type,str(target_id),json.dumps(payload or {}),priority,NOW(),NOW(),NOW()))

    def paper(self,doi,title,year=None,pmid=None,pmcid=None):
        doi=(doi or '').strip().lower()
        if doi.startswith('https://doi.org/'):doi=doi[len('https://doi.org/'):]
        row=self.rows('SELECT paper_id FROM papers WHERE (doi=? AND doi IS NOT NULL) OR (pmid=? AND pmid IS NOT NULL) OR (pmcid=? AND pmcid IS NOT NULL)',(doi or None,pmid,pmcid))
        if len(row)>1:raise ReviewError('CONFLICTING_PAPER_IDENTIFIERS')
        if row:return row[0]['paper_id']
        pid=self.run('INSERT INTO papers (doi,pmid,pmcid,title,publication_year,landing_url,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)',
                     (doi or None,pmid,pmcid,title,year,'https://doi.org/'+doi if doi else 'https://pubmed.ncbi.nlm.nih.gov/'+str(pmid)+'/',NOW(),NOW()))
        atomic_json(ROOT/'papers'/f'p{pid:06d}'/'paper.json',self.rows('SELECT * FROM papers WHERE paper_id=?',(pid,))[0])
        return pid

    def dataset(self,pid,url,description,source_key=None):
        key=source_key or url
        row=self.rows('SELECT dataset_id FROM datasets WHERE source_key=?',(key,))
        if row:did=row[0]['dataset_id']
        else:
            did=self.run('INSERT INTO datasets (source_key,repository,version,landing_url,description,collection_decision,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)',
                         (key,urllib.parse.urlsplit(url).hostname or 'source','observed:'+NOW()[:10],url,description[:1500],'download_candidate',NOW(),NOW()))
        if pid is not None:
            self.run('INSERT OR IGNORE INTO paper_datasets VALUES (?,?,?,?,?,?)',
                     (pid,did,'unknown',url,f'papers/p{pid:06d}/evidence',description[:1500]))
        return did

    def file(self,did,url,name,size=None,published_sha=None,source_key=None):
        if not isinstance(size,int):size=None
        key=source_key or url
        row=self.rows('SELECT file_id FROM files WHERE source_key=?',(key,))
        if row:fid=row[0]['file_id']
        else:
            fid=self.run('INSERT INTO files (source_key,landing_url,download_url,original_name,role,declared_bytes,published_checksum_algorithm,published_checksum) VALUES (?,?,?,?,?,?,?,?)',
                         (key,url,url,name,'data',size,'sha256' if published_sha else None,published_sha))
        self.run('INSERT OR IGNORE INTO dataset_files VALUES (?,?)',(did,fid))
        if size is not None and size>536870912:
            self.run("UPDATE files SET file_state='review' WHERE file_id=?",(fid,));return fid
        self.enqueue('download','file',fid,priority=30 if size is not None and size<5_000_000 else 50)
        return fid

    def seeds(self):
        for seed in json.loads((ROOT/'config/seeds.json').read_text())['papers']:
            pid=self.paper(seed['doi'],seed['title'],seed['year'])
            self.run("UPDATE papers SET screening_state='candidate',reason_code='PREVIOUSLY_VERIFIED_SOURCE' WHERE paper_id=?",(pid,))
            did=self.dataset(pid,seed['dataset_landing_url'],'Previously downloaded and audited by user assistant on 2026-09-15; fresh file verification required.',source_key='seed:'+seed['legacy_paper_key'])
            for f in seed['previously_downloaded_files']:
                if f.get('observation_role')=='derived_analysis_not_additional_observations':continue
                self.file(did,f['url_observed'],f['original_name'],f['bytes_previously_observed'],f['sha256_previously_observed'])
        extra=ROOT/'config/extra_neural_sources.json'
        if extra.is_file():
            for source in json.loads(extra.read_text())['datasets']:
                did=self.dataset(None,source['landing_url'],source['title'],source['source_key'])
                self.run("UPDATE datasets SET species_status='yes',version=?,license_url=?,metadata_json=? WHERE dataset_id=?",(source['version'],source.get('license_url'),json.dumps(source,ensure_ascii=False),did))
                for item in source['files']:
                    fid=self.file(did,item['url'],item['name'],item.get('size'),source_key=item['source_key'])
                    if item.get('role'):self.run('UPDATE files SET role=? WHERE file_id=?',(item['role'],fid))
                    self.run("UPDATE jobs SET priority=15 WHERE action='download' AND target_id=? AND status='queued'",(str(fid),))
        for row in self.rows("SELECT file_id FROM files WHERE file_state='verified' AND role='data'"):
            self.enqueue('parse','file',row['file_id'],priority=10)

    def discover(self):
        cycle=self.kv('discovery_cycle',0)
        scope='neural' if cycle%3!=2 else 'behavior'
        cursor_key='pubmed_neural_cursor' if scope=='neural' else 'pubmed_cursor'
        cursor=self.kv(cursor_key,{'year':2026,'offset':0})
        year=cursor['year'];offset=cursor['offset']
        term='("Caenorhabditis elegans"[Title/Abstract] OR "C. elegans"[Title/Abstract]) AND (behavior*[Title/Abstract] OR behaviour*[Title/Abstract] OR locomotion[Title/Abstract] OR posture[Title/Abstract] OR eigenworm*[Title/Abstract] OR tracking[Title/Abstract])'
        if scope=='neural':
            term='("Caenorhabditis elegans"[Title/Abstract] OR "C. elegans"[Title/Abstract]) AND (calcium[Title/Abstract] OR neuron*[Title/Abstract] OR neural[Title/Abstract] OR brain-wide[Title/Abstract]) AND (imaging[Title/Abstract] OR activit*[Title/Abstract] OR recording*[Title/Abstract] OR dynamics[Title/Abstract])'
        query={'db':'pubmed','term':term,'retmode':'json','retmax':10,'retstart':offset,'datetype':'pdat','mindate':f'{year}/01/01','maxdate':f'{year}/12/31','sort':'pub date','tool':'worm_behavior_collector'}
        self.log('pubmed_search_started',scope=scope,year=year,offset=offset)
        base='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
        answer=self.fetch_json(base+'esearch.fcgi?'+urllib.parse.urlencode(query))
        result=answer['esearchresult'];ids=result.get('idlist',[])
        total=int(result.get('count',0))
        if ids:
            raw=self.fetch(base+'efetch.fcgi?'+urllib.parse.urlencode({'db':'pubmed','id':','.join(ids),'retmode':'xml'}),limit=5_000_000)
            tree=ET.fromstring(raw)
            for article in tree.findall('.//PubmedArticle'):
                aid=pubmed_article_ids(article)
                pmid=xmltext(article.find('.//MedlineCitation/PMID')); title=xmltext(article.find('.//ArticleTitle'))
                if not pmid or not title:continue
                pid=self.paper(aid.get('doi'),title,year,pmid,aid.get('pmc'))
                abstract=' '.join(xmltext(x) for x in article.findall('.//AbstractText'))
                payload={'pmid':pmid,'pmcid':aid.get('pmc'),'title':title,'abstract':abstract,'discovery_scope':scope}
                atomic_json(ROOT/'papers'/f'p{pid:06d}'/'evidence'/'pubmed.json',payload)
                self.enqueue('screen','paper',pid,payload,priority=20)
        cursor={'year':year,'offset':offset+len(ids)}
        if not ids or offset+len(ids)>=min(total,9990):cursor={'year':year-1 if year>1990 else 2026,'offset':0}
        self.setkv(cursor_key,cursor);self.setkv('last_discovery_at',NOW());self.setkv('discovery_cycle',cycle+1)
        self.log('pubmed_search_completed',scope=scope,year=year,total_in_year=total,returned=len(ids),next_cursor=cursor)

    def evidence(self,pid,payload):
        title=payload.get('title','');abstract=payload.get('abstract','')
        text=title+'\n'+abstract;links=[]
        pmcid=payload.get('pmcid')
        if pmcid:
            try:
                raw=self.fetch('https://www.ebi.ac.uk/europepmc/webservices/rest/'+pmcid+'/fullTextXML',limit=6_000_000)
                tree=ET.fromstring(raw)
                sections=[s for s in tree.iter('sec') if any(w in (s.get('sec-type','')+' '+xmltext(s.find('title'))).lower() for w in ['data avail','availability','data-access','data and code'])]
                source_nodes=sections or [tree]
                texts=[]
                for node in source_nodes:
                    texts.append(xmltext(node)[:18000])
                    for link in node.iter('ext-link'):
                        url=link.get('{http://www.w3.org/1999/xlink}href','')
                        if self.repository_url(url) and url not in [x['url'] for x in links]:
                            links.append({'link_id':f'L{len(links)+1}','url':url,'label':xmltext(link)[:150],'availability_section':bool(sections)})
                text+='\n'+'\n'.join(texts)[:8000]
            except (urllib.error.URLError,ValueError,ReviewError,ET.ParseError):pass
        # Repository URLs present in abstracts can still be reviewed, but not auto-resolved.
        for url in re.findall(r'https?://[^\s<>"\]]+',text):
            url=html.unescape(url).rstrip('.,;)')
            if self.repository_url(url) and url not in [x['url'] for x in links]:
                links.append({'link_id':f'L{len(links)+1}','url':url,'label':'source text URL','availability_section':False})
        value={'candidate_id':str(pid),'title':title,'evidence_blocks':[{'block_id':'B1','text':text[:10000]}],'links':links[:20]}
        atomic_json(ROOT/'papers'/f'p{pid:06d}'/'evidence'/'candidate.json',value)
        return value

    @staticmethod
    def repository_url(url):
        try:host=(urllib.parse.urlsplit(url).hostname or '').lower()
        except ValueError:return False
        return any(host==h or host.endswith('.'+h) for h in REPOSITORIES)

    def screen(self,job):
        k='llm_calls:'+NOW()[:10]
        if not self.model_budget_available():raise BudgetError('DAILY_MODEL_BUDGET')
        pid=int(job['target_id']);candidate=self.evidence(pid,json.loads(job['payload_json']))
        key=pathlib.Path('/etc/worm-collector/deepseek.key').read_text().strip()
        prompt=(ROOT/'config/AGENT_PROMPT.md').read_text()+'\n\nJSON output schema:\n'+(ROOT/'config/decision.schema.json').read_text()
        payload={'model':'deepseek-flash','thinking':{'type':'disabled'},'temperature':0,'max_tokens':700,
                 'response_format':{'type':'json_object'},'messages':[{'role':'system','content':prompt},{'role':'user','content':json.dumps(candidate,ensure_ascii=False)}]}
        for attempt in range(2):
            if not self.model_budget_available():raise BudgetError('DAILY_MODEL_BUDGET')
            self.setkv(k,self.kv(k,0)+1)
            response=self.fetch_json('https://api.deepseek.com/chat/completions',limit=200000,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            content=response['choices'][0]['message']['content']
            try:
                decision=json.loads(content);self.validate_decision(candidate,decision);break
            except (ReviewError,ValueError,TypeError) as e:
                if attempt:raise
                payload['messages']+= [{'role':'assistant','content':content[:6000]},
                    {'role':'user','content':'JSON validation failed: '+type(e).__name__+'. Return corrected JSON using the provided schema. Evidence quotes must be exact SHORT substrings (at most 160 characters) from B1. Unknown evidence must be unknown/review. Do not invent any links or facts.'}]
        atomic_json(ROOT/'papers'/f'p{pid:06d}'/'decision.json',decision)
        state={'download_candidate':'candidate','metadata_only':'review','exclude':'excluded','review':'review'}[decision['decision']]
        self.run('UPDATE papers SET screening_state=?,reason_code=?,evidence_text=?,updated_at=? WHERE paper_id=?',
                 (state,decision['reason_code'],json.dumps(decision['evidence_quotes'],ensure_ascii=False),NOW(),pid))
        selected={x['link_id']:x for x in candidate['links']}
        resolved=0
        if decision['decision']=='download_candidate':
            for lid in decision['selected_link_ids']:
                link=selected[lid]
                if not link['availability_section']:continue
                did=self.dataset(pid,link['url'],json.dumps(decision['evidence_quotes'],ensure_ascii=False))
                self.enqueue('resolve','dataset',did,{'url':link['url']},priority=25);resolved+=1
        self.log('candidate_screened',paper_id=pid,decision=decision['decision'],neural_series=decision.get('measured_neural_series','unknown'),behavior_series=decision['measured_behavior_series'],repository_tasks=resolved,model='deepseek-flash')

    @staticmethod
    def validate_decision(candidate,d):
        required={'schema_version','candidate_id','species','measured_behavior_series','representation','access','decision','reason_code','selected_link_ids','evidence_quotes'}
        if isinstance(d,dict) and d.get('schema_version')=='2.0':required.add('measured_neural_series')
        if not isinstance(d,dict) or set(d)!=required or d['schema_version'] not in ('1.0','2.0') or d['candidate_id']!=candidate['candidate_id']:raise ReviewError('MODEL_SCHEMA')
        if d['decision'] not in ('download_candidate','metadata_only','exclude','review'):raise ReviewError('MODEL_DECISION')
        for k in ('species','measured_behavior_series'):
            if d[k] not in ('yes','no','unknown'):raise ReviewError('MODEL_ENUM')
        neural=d.get('measured_neural_series','unknown')
        if neural not in ('yes','no','unknown'):raise ReviewError('MODEL_ENUM')
        relevant=d['measured_behavior_series']=='yes' or neural=='yes'
        if d['representation'] not in ('kinematics','eigenworm','body_shape','position2d','mixed','unknown','neural_calcium','neural_voltage','neural_spikes'):raise ReviewError('MODEL_ENUM')
        if d['access'] not in ('public_processed','public_raw_video','restricted','unknown'):raise ReviewError('MODEL_ENUM')
        if not isinstance(d['selected_link_ids'],list) or len(d['selected_link_ids'])>5 or len(set(d['selected_link_ids']))!=len(d['selected_link_ids']):raise ReviewError('MODEL_LINKS')
        if not set(d['selected_link_ids'])<={x['link_id'] for x in candidate['links']}:raise ReviewError('INVENTED_LINK')
        quotes=d['evidence_quotes'];blocks={x['block_id']:x['text'] for x in candidate['evidence_blocks']}
        if not isinstance(quotes,list) or len(quotes)>3:raise ReviewError('MODEL_QUOTES')
        for q in quotes:
            if not isinstance(q,dict) or set(q)!={'block_id','quote'} or not isinstance(q['quote'],str) or not 1<=len(q['quote'])<=160 or q['quote'] not in blocks.get(q['block_id'],''):raise ReviewError('UNGROUNDED_QUOTE')
        reasons={'download_candidate':['ELIGIBLE_SERIES'],'metadata_only':['RAW_VIDEO_ONLY','RESTRICTED_ACCESS','NO_DATA_LINK','OVER_BUDGET'],'exclude':['WRONG_SPECIES','NO_BEHAVIOR_SERIES','NO_RELEVANT_SERIES','SIMULATION_ONLY','SUMMARY_ONLY'],'review':['INSUFFICIENT_EVIDENCE','CONFLICTING_EVIDENCE']}
        if d['reason_code'] not in reasons[d['decision']]:raise ReviewError('MODEL_REASON')
        if d['decision']=='download_candidate' and not(d['species']=='yes' and relevant and d['access']=='public_processed' and d['selected_link_ids'] and quotes):raise ReviewError('MODEL_CONTRADICTION')
        if d['decision']=='exclude' and (not quotes or (d['reason_code']=='WRONG_SPECIES' and d['species']!='no') or (d['reason_code']!='WRONG_SPECIES' and (d['measured_behavior_series']!='no' or neural=='yes'))):raise ReviewError('MODEL_CONTRADICTION')
        if d['schema_version']=='2.0' and d['decision']=='exclude' and d['reason_code']!='WRONG_SPECIES' and neural!='no':raise ReviewError('MODEL_CONTRADICTION')

    def resolve(self,job):
        did=int(job['target_id']);url=json.loads(job['payload_json'])['url'];p=urllib.parse.urlsplit(url);host=p.hostname
        if host=='zenodo.org':
            match=re.search(r'/(?:records?|record)/(\d+)',p.path)
            if not match:raise ReviewError('ZENODO_RECORD_ID_MISSING')
            record=self.fetch_json('https://zenodo.org/api/records/'+match.group(1),limit=4_000_000)
            atomic_json(ROOT/'datasets'/f'd{did:06d}'/'repository.json',record)
            files=record.get('files',[])
            for f in files[:100]:
                name=f.get('key','');link=f.get('links',{}).get('self') or f.get('links',{}).get('download')
                if link and name.lower().endswith(SUFFIXES):self.file(did,link,name,f.get('size'),source_key=f'zenodo:{record["id"]}:{name}')
            self.log('repository_resolved',dataset_id=did,repository='zenodo',listed_files=len(files))
        elif host=='github.com':
            parts=p.path.strip('/').split('/')
            if len(parts)<2:raise ReviewError('GITHUB_REPOSITORY_MISSING')
            repo='/'.join(parts[:2]);base='https://api.github.com/repos/'+repo
            meta=self.fetch_json(base);branch=meta['default_branch']
            commit=self.fetch_json(base+'/commits/'+urllib.parse.quote(branch,safe=''))
            sha=commit['sha']
            tree=self.fetch_json(base+'/git/trees/'+sha+'?recursive=1',limit=4_000_000)
            count=0
            for f in tree.get('tree',[]):
                if f.get('type')=='blob' and f['path'].lower().endswith(SUFFIXES) and not any(x in f['path'].lower().split('/') for x in ['test','tests','node_modules','.github']):
                    if not any(x in f['path'].lower() for x in ['data','trace','worm','behavior','posture','neural','calcium','gcamp']):continue
                    raw='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+urllib.parse.quote(f['path'])
                    self.file(did,raw,f['path'],f.get('size'),source_key='github:'+repo+':'+sha+':'+f['path']);count+=1
                    if count>=20:break
            if not count:raise ReviewError('NO_BOUNDED_DATA_FILES')
            self.log('repository_resolved',dataset_id=did,repository='github',listed_files=count)
        elif host=='datadryad.org':
            match=re.search(r'10\.5061/[^?#]+',urllib.parse.unquote(url))
            if not match:raise ReviewError('DRYAD_DOI_MISSING')
            meta=self.fetch_json('https://datadryad.org/api/v2/datasets/'+urllib.parse.quote('doi:'+match.group(0),safe=''),limit=4_000_000)
            link=meta.get('_links',{}).get('stash:version',{}).get('href')
            if not link:raise ReviewError('DRYAD_VERSION_MISSING')
            version=self.fetch_json(urllib.parse.urljoin('https://datadryad.org',link))
            files_url=version.get('_links',{}).get('stash:files',{}).get('href')
            if not files_url:raise ReviewError('DRYAD_FILES_MISSING')
            listing=self.fetch_json(urllib.parse.urljoin('https://datadryad.org',files_url),limit=4_000_000)
            files=listing.get('_embedded',{}).get('stash:files',[])
            for f in files[:100]:
                name=f.get('path') or f.get('name') or f.get('filename','')
                href=f.get('_links',{}).get('stash:download',{}).get('href')
                if href and name.lower().endswith(SUFFIXES):self.file(did,urllib.parse.urljoin('https://datadryad.org',href),name,f.get('size'),source_key='dryad-file:'+str(f.get('id',href)))
            self.log('repository_resolved',dataset_id=did,repository='dryad',listed_files=len(files))
        else:raise ReviewError('REPOSITORY_ADAPTER_PENDING')

    def heartbeat(self,job):
        self.run("UPDATE jobs SET heartbeat_at=?,lease_until=?,updated_at=? WHERE job_id=? AND lease_token=? AND status='running'",(NOW(),self.future(900),NOW(),job['job_id'],job['lease_token']))

    @staticmethod
    def future(seconds):return (dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds)).strftime('%Y-%m-%dT%H:%M:%SZ')

    def download(self,job):
        fid=int(job['target_id']);f=self.rows('SELECT * FROM files WHERE file_id=?',(fid,))[0]
        if f['file_state']=='verified' and f['object_relpath'] and (ROOT/f['object_relpath']).is_file():return
        url=f['download_url'];host=urllib.parse.urlsplit(url).hostname
        if self.kv('host_backoff:'+host,0)>time.time():raise ReviewError('HOST_BACKOFF')
        with self.lock:hostlock=self.hostlocks.setdefault(host,threading.Lock())
        with hostlock:
            maximum=536870912
            if f['declared_bytes'] is not None and f['declared_bytes']>maximum:raise ReviewError('FILE_OVER_CAP')
            self.reserve(f['declared_bytes'] or maximum)
            reservation=f['declared_bytes'] or maximum;written=0;part=ROOT/'incoming'/f'{job["job_id"]}_{job["lease_token"]}.part'
            try:
                with self.open(url) as response:
                    announced=int(response.headers.get('Content-Length','0') or 0)
                    if announced>maximum:raise ReviewError('FILE_OVER_CAP')
                    content_type=response.headers.get('Content-Type','')
                    self.log('download_started',file_id=fid,name=f['original_name'],announced_bytes=announced)
                    h=hashlib.sha256();first=b'';lastbeat=time.monotonic();start=time.monotonic()
                    with part.open('wb') as out:
                        while True:
                            if STOP.is_set():raise BudgetError('STOPPING')
                            if time.monotonic()-start>3600:raise TimeoutError('DOWNLOAD_TIME_LIMIT')
                            block=response.read(65536)
                            if not block:break
                            if not first:first=block[:1024]
                            if written+len(block)>maximum or written+len(block)>reservation:raise ReviewError('FILE_LENGTH_CHANGED')
                            self.charge(len(block));out.write(block);h.update(block);written+=len(block)
                            # At most 256 KiB/s per worker; at most 512 KiB/s aggregate.
                            delay=written/262144-(time.monotonic()-start)
                            if delay>0 and STOP.wait(delay):raise BudgetError('STOPPING')
                            if time.monotonic()-lastbeat>25:self.heartbeat(job);lastbeat=time.monotonic()
                        out.flush();os.fsync(out.fileno())
                    if announced and written!=announced:raise ReviewError('TRUNCATED_TRANSFER')
                    if not written or first.lstrip().lower().startswith((b'<!doctype html',b'<html')):raise ReviewError('NOT_DATA_HTML_OR_EMPTY')
                    suffix=f['original_name'].lower()
                    signatures=[(('.zip','.npz'),(b'PK\x03\x04',b'PK\x05\x06')),(('.bz2',),(b'BZh',)),(('.gz',),(b'\x1f\x8b',)),(('.mat',),(b'MATLAB',b'\x89HDF')),(('.h5','.hdf5','.nwb'),(b'\x89HDF',b'MATLAB')),(('.npy',),(b'\x93NUMPY',))]
                    for endings,magic in signatures:
                        if suffix.endswith(endings) and not first.startswith(magic):raise ReviewError('FILE_SIGNATURE_MISMATCH')
                    if suffix.endswith('.json') and not first.lstrip().startswith((b'{',b'[')):raise ReviewError('FILE_SIGNATURE_MISMATCH')
                    sha=h.hexdigest()
                    if f['published_checksum_algorithm']=='sha256' and f['published_checksum']!=sha:raise ReviewError('SOURCE_SNAPSHOT_CHECKSUM_CHANGED')
                    lease=self.rows("SELECT lease_token,lease_until,status FROM jobs WHERE job_id=?",(job['job_id'],))[0]
                    if lease['lease_token']!=job['lease_token'] or lease['status']!='running' or lease['lease_until']<NOW():raise ReviewError('LOST_LEASE')
                    rel='objects/sha256/'+sha[:2]+'/'+sha;dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        existing=hashlib.sha256()
                        with dest.open('rb') as existing_file:
                            for b in iter(lambda:existing_file.read(1048576),b''):existing.update(b)
                        if existing.hexdigest()!=sha:raise ReviewError('EXISTING_OBJECT_CORRUPT')
                        part.unlink();written_for_space=0
                    else:os.replace(part,dest);written_for_space=written
                    self.run("UPDATE files SET file_state='verified',downloaded_bytes=?,sha256=?,object_relpath=?,published_checksum_matches=?,content_type_observed=?,downloaded_at=?,verified_at=? WHERE file_id=?",(written,sha,rel,1 if f['published_checksum'] else None,content_type,NOW(),NOW(),fid))
                    atomic_json(ROOT/'datasets'/'receipts'/f'f{fid:06d}.json',{'file_id':fid,'sha256':sha,'bytes':written,'at':NOW(),'validation_level':'transport_length_and_local_sha256_plus_published_checksum_when_available','source_name':f['original_name'],'numeric_data_status':'not_parsed','object_relpath':rel})
                    self.enqueue('parse','file',fid,priority=90)
                    self.log('download_verified',file_id=fid,bytes=written,sha256=sha)
                    self.release(reservation,written_for_space);reservation=0
            finally:
                if part.exists():
                    # Only this job-owned temporary file is removed; verified objects are never deleted.
                    part.unlink()
                if reservation:self.release(reservation)

    def claim(self,actions):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            params=tuple(actions)+(NOW(),)
            row=self.db.execute('SELECT * FROM jobs WHERE action IN ('+','.join('?' for _ in actions)+") AND status IN ('queued','retry_wait') AND next_run_at<=? AND attempts<max_attempts ORDER BY priority,job_id LIMIT 1",params).fetchone()
            if not row:self.db.commit();return None
            j=dict(row);j['lease_token']=uuid.uuid4().hex
            self.db.execute("UPDATE jobs SET status='running',attempts=attempts+1,lease_token=?,lease_until=?,heartbeat_at=?,updated_at=? WHERE job_id=?",(j['lease_token'],self.future(900),NOW(),NOW(),j['job_id']))
            self.db.commit();j['attempts']+=1;return j

    def execute(self,job):
        status='done';error=None;nexttime=NOW();decrement=0
        try:
            if job['action']=='screen':self.screen(job)
            elif job['action']=='resolve':self.resolve(job)
            elif job['action']=='download':self.download(job)
            elif job['action']=='parse':
                from neural_data import parse_file,DataReview
                try:parse_file(self,job)
                except DataReview as e:raise ReviewError(str(e))
                except InterruptedError:raise BudgetError('STOPPING')
        except BudgetError as e:
            status='retry_wait';error=str(e);nexttime=self.future(3600);decrement=1
        except ReviewError as e:
            status='review';error=str(e)
        except urllib.error.HTTPError as e:
            error='HTTP_'+str(e.code)
            status='review' if e.code in (400,401,403,404) else 'retry_wait'
            nexttime=self.future(3600 if job['attempts']>1 or e.code==429 else 300)
        except (urllib.error.URLError,TimeoutError,ConnectionError,OSError):
            status='retry_wait';error='NETWORK_OR_IO_ERROR';nexttime=self.future(300 if job['attempts']==1 else 3600)
        except (KeyError,ValueError,TypeError,ET.ParseError):
            status='review';error='SOURCE_OR_MODEL_SCHEMA'
        except Exception as e:
            status='review';error='UNEXPECTED_'+type(e).__name__
        if status=='retry_wait' and not decrement and job['attempts']>=job['max_attempts']:status='failed'
        self.run('UPDATE jobs SET status=?,attempts=attempts-?,last_error_code=?,next_run_at=?,lease_token=NULL,lease_until=NULL,reserved_bytes=0,updated_at=? WHERE job_id=? AND lease_token=?',
                 (status,decrement,error,nexttime,NOW(),job['job_id'],job['lease_token']))
        if status in ('review','failed'):
            if job['action']=='download':self.run("UPDATE files SET file_state='review' WHERE file_id=? AND file_state<>'verified'",(job['target_id'],))
            atomic_json(ROOT/'review'/f'j{job["job_id"]:06d}.json',{'job_id':job['job_id'],'action':job['action'],'target':job['target_id'],'state':status,'reason':error,'at':NOW()})
        self.log('job_finished',job_id=job['job_id'],action=job['action'],status=status,error=error)

    def publish(self):
        stamp=NOW().replace(':','').replace('-','');directory=ROOT/'exports'/('snapshot_'+stamp);directory.mkdir(parents=True,exist_ok=True)
        with self.lock:
            self.db.execute('BEGIN')
            names=['papers','datasets','files','recordings','channels','jobs']
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='neural_recordings'").fetchone():names+=['neural_recordings','neural_record_sources','neural_neurons']
            tables={name:[dict(r) for r in self.db.execute('SELECT * FROM '+name)] for name in names}
            self.db.commit()
        for name,rows in tables.items():
            if rows:
                with (directory/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            else:
                columns=[r['name'] for r in self.rows('PRAGMA table_info('+name+')')]
                with (directory/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:csv.writer(f).writerow(columns)
        objects={f['sha256']:f['downloaded_bytes'] for f in tables['files'] if f['file_state']=='verified'}
        neural=tables.get('neural_recordings',[])
        summary={'at':NOW(),'version':PROGRAM_VERSION,'phase':'behavior_and_neural_collection_with_neural_standardization',
                 'papers':len(tables['papers']),'datasets':len(tables['datasets']),'verified_files':sum(f['file_state']=='verified' for f in tables['files']),
                 'unique_downloaded_bytes':sum(objects.values()),'training_ready_recordings':sum(r['training_eligible'] for r in neural),
                 'standardized_neural_recordings':len(neural),'neural_timepoints':sum(r['n_timepoints'] for r in neural),
                 'named_neuron_observations':sum(r['n_named_neurons'] for r in neural),
                 'training_scope':'within-record chronological forecasting; cross-animal identity review required',
                 'job_states':{s:sum(j['status']==s for j in tables['jobs']) for s in ['queued','running','retry_wait','done','review','failed']},
                 'daily_transfer_bytes':self.kv('transfer:'+NOW()[:10],0),'daily_model_calls':self.kv('llm_calls:'+NOW()[:10],0),
                 'daily_llm_calls_max':self.policy.get('daily_llm_calls_max',300),
                 'last_discovery_at':self.kv('last_discovery_at'),'pubmed_cursor':self.kv('pubmed_cursor'),
                 'pubmed_neural_cursor':self.kv('pubmed_neural_cursor'),'daily_transfer_max_bytes':self.policy.get('daily_transfer_max_bytes'),
                 'supported_neural_parsers':['wormwideweb-json','dryad-copper-hdf5'],'unsupported_formats_require_review':True}
        atomic_json(ROOT/'state/status.json',summary);atomic_json(ROOT/'exports/current.json',{'path':str(directory.relative_to(ROOT))})
        snapshots=sorted((ROOT/'exports').glob('snapshot_*'))
        for old in snapshots[:-7]:
            if old.is_dir() and old.resolve().parent==(ROOT/'exports').resolve():shutil.rmtree(old)
        self.log('catalog_published',papers=summary['papers'],verified_files=summary['verified_files'])

    def tick(self):
        settings=ROOT/'state/control.json'
        if settings.exists() and not json.loads(settings.read_text()).get('enabled',False):return
        self.log('tick_started',pid=os.getpid())
        # The OS flock proves no prior collector remains; recover its interrupted tasks.
        self.run("UPDATE jobs SET status=CASE WHEN attempts>=max_attempts THEN 'failed' ELSE 'retry_wait' END,lease_token=NULL,lease_until=NULL,last_error_code='PREVIOUS_PROCESS_INTERRUPTED',next_run_at=?,updated_at=? WHERE status='running'",(NOW(),NOW()))
        self.run("UPDATE jobs SET next_run_at=?,updated_at=? WHERE status='retry_wait' AND last_error_code='STOPPING'",(NOW(),NOW()))
        self.seeds()
        # Continue backfill while the bounded screening queue and daily budget allow.
        for part in (ROOT/'incoming').glob('*.part'):
            if re.fullmatch(r'\d+_[0-9a-f]{32}\.part',part.name):
                size=part.stat().st_size;part.unlink();self.occupied=max(0,self.occupied-size)
        last=self.kv('last_discovery_epoch',0)
        waiting=self.rows("SELECT count(*) AS n FROM jobs WHERE action='screen' AND status IN ('queued','retry_wait')")[0]['n']
        model_budget=self.model_budget_available()
        if time.time()-last>60 and waiting<15 and model_budget:
            try:self.discover();self.setkv('last_discovery_epoch',time.time())
            except Exception as e:self.log('discovery_error',error=type(e).__name__,detail=str(e)[:180]);self.setkv('last_discovery_error',type(e).__name__)
        self.publish()
        # One remote-model worker; two downloads in parallel, same-host serialized.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            active={}
            launched=0;screened=0;parsed=0
            while time.monotonic()<self.deadline and not STOP.is_set():
                for future,job in list(active.items()):
                    if future.done():future.result();del active[future];self.publish()
                while len(active)<2 and launched<12:
                    j=self.claim(['download'])
                    if not j:break
                    active[pool.submit(self.execute,j)]=j;launched+=1
                if parsed<20 and not STOP.is_set():
                    j=self.claim(['parse'])
                    if j:self.execute(j);parsed+=1;self.publish();continue
                if screened<30:
                    j=self.claim(['resolve']) or self.claim(['screen'])
                    if j:self.execute(j);screened+=1;self.publish();continue
                if not active:break
                STOP.wait(2)
            if active:self.log('waiting_for_owned_downloads',count=len(active))
        for _ in range(12):
            j=self.claim(['parse'])
            if not j:break
            self.execute(j)
        self.publish();self.log('tick_finished')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','tick','status','init']);args=ap.parse_args()
    if args.action=='status':
        p=ROOT/'state/status.json';print(p.read_text() if p.exists() else '{"status":"not_started"}');return
    (ROOT/'state').mkdir(parents=True,exist_ok=True)
    with (ROOT/'state/tick.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:STOP.set())
        while not STOP.is_set():
            agent=Agent()
            try:
                if args.action=='init':agent.seeds();agent.publish()
                else:agent.tick()
            finally:agent.db.close()
            if args.action!='run':break
            # This is a resident process, not cron. Rate/budget pauses avoid busy loops.
            STOP.wait(15)

if __name__=='__main__':main()
