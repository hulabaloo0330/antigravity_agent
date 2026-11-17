"""Versioned, bounded numerical adapters. Never execute code from a data source."""
import bz2, csv, hashlib, json, math, os, pathlib, re, shutil, tarfile, time, uuid
import numpy as np

PARSER_VERSION='neural-v2.0'
MAX_JSON=64*1024*1024
MAX_CELLS=5_000_000
BEHAVIOR_KEYS=('velocity','head_angle','angular_velocity','pumping')

class DataReview(Exception):pass

def sha_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def jwrite(path,value):
    with pathlib.Path(path).open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value

def check_time(values):
    t=np.asarray(values,dtype='<f8')
    if t.ndim!=1 or len(t)<2 or len(t)>200_000 or not np.isfinite(t).all():raise DataReview('INVALID_TIME_AXIS')
    d=np.diff(t)
    if not (d>0).all():raise DataReview('NONMONOTONIC_TIME')
    median=float(np.median(d));ends=[0]+(np.flatnonzero(d>1.5*median)+1).tolist()+[len(t)]
    return t,median,[{'start':int(a),'stop':int(b)} for a,b in zip(ends[:-1],ends[1:])]

def prepare_www(obj):
    if not isinstance(obj,dict) or not all(k in obj for k in ('metadata','timing','gcamp')):raise DataReview('NOT_WORMWIDEWEB_NEURAL_JSON')
    md=obj['metadata'];g=obj['gcamp'];tm=obj['timing']
    if not isinstance(md,dict) or not md.get('uid'):raise DataReview('SOURCE_ID_MISSING')
    t,dt,segments=check_time(tm['timestamp_confocal'])
    field='trace_array_original' if g.get('trace_array_original') is not None else 'trace_array'
    a=np.asarray(g[field],dtype='<f4')
    if a.ndim!=2 or a.shape[1]!=len(t) or not 1<=a.shape[0]<=4096 or a.size>MAX_CELLS:raise DataReview('NEURAL_TIME_SHAPE_MISMATCH')
    if md.get('n_neuron') is not None and md['n_neuron']!=a.shape[0]:raise DataReview('NEURON_COUNT_MISMATCH')
    signal=np.ascontiguousarray(a.T);valid=np.isfinite(signal)
    if not valid.any():raise DataReview('NO_FINITE_NEURAL_SIGNAL')
    neurons=[];labels=obj.get('label') or {}
    for j in range(a.shape[0]):
        label=labels.get(str(j+1),{}) or {}
        if not isinstance(label,dict):label={}
        name=label.get('label')
        if not isinstance(name,str) or not name.strip() or name.lower() in ('unknown','unidentified','undefined','none'):name=None
        vals=signal[:,j][valid[:,j]]
        neurons.append({'column_index':j,'source_row_index':j+1,'local_channel_id':'roi_%04d'%(j+1),
            'neuron_name':name,'neuron_class':label.get('neuron_class'),'confidence_source':label.get('confidence'),
            'source_label':clean(label),'n_finite':int(len(vals)),'is_constant':bool(len(vals)>0 and np.ptp(vals)==0)})
    author=None
    if field=='trace_array_original' and g.get('trace_array') is not None:
        candidate=np.asarray(g['trace_array'],dtype='<f4')
        if candidate.shape==a.shape:author=np.ascontiguousarray(candidate.T)
    b=obj.get('behavior') or {};channels=[];arrays=[];issues=[]
    for key in BEHAVIOR_KEYS:
        value=b.get(key)
        if value is None:continue
        arr=np.asarray(value,dtype='<f4')
        if arr.ndim!=1 or len(arr)!=len(t):issues.append('BEHAVIOR_LENGTH_MISMATCH:'+key);continue
        # Preserve source scale. Velocity in WWW JSON is 0.1 mm/s; no hidden conversion.
        unit={'velocity':'0.1 mm/s','head_angle':'rad','angular_velocity':'rad/s','pumping':'1/s'}[key]
        channels.append({'name':key,'source_unit':unit,'conversion':'identity'});arrays.append(arr)
    condition=md.get('dataset_type') or []
    if isinstance(condition,str):condition=[condition]
    is_control=any('gfp' in str(x).lower() for x in condition)
    metadata={'source_format':'wormwideweb-json','source_record_id':str(md['uid']),'source_metadata':clean(md),
        'source_group_id':'source-h5:'+str(md['checksum_h5']) if md.get('checksum_h5') else 'source-session:'+str(md.get('paper_id'))+':'+str(md['uid']),
        'animal_id':None,'animal_identity_status':'unknown','condition':condition,
        'signal_kind':'GFP_control_fluorescence' if is_control else 'calcium_reporter_fluorescence',
        'signal_source_field':'gcamp/'+field,'signal_unit':'source_fluorescence_scale',
        'source_preprocessing':'Author-provided extracted fluorescence; not raw photons or membrane voltage. Upstream processing is not proven causal.',
        'normalization_applied_here':'none','source_author_zscore_uses_future_statistics':'possible; excluded by default loader',
        'time_unit':'s','time_unit_evidence':'WormWideWeb JSON schema: timestamp_confocal and mean_timestep',
        'source_time_origin':float(t[0]),'event_index_origin':'unknown_preserved_not_converted',
        'events_source_index':clean(tm.get('event',{})),'reversal_events_source_index':clean(b.get('reversal_events',[])),
        'behavior_channels':channels,'issues':issues+['UPSTREAM_CAUSAL_PREPROCESSING_NOT_VERIFIED'],
        'source_original_available':field=='trace_array_original','control':is_control}
    behavior=np.column_stack(arrays).astype('<f4') if arrays else None
    return {'signal':signal,'valid':valid,'time_source':t,'time_s':t-t[0],'median_dt_s':dt,'segments':segments,
        'neurons':neurons,'author_zscore':author,'behavior':behavior,'metadata':metadata}

def prepare_copper_h5(path,source_name):
    """Only the author-documented Dryad copper layout; not arbitrary HDF5 guessing."""
    import h5py
    with h5py.File(path,'r') as f:
        required=('gcamp/trace_array_original','timing/timestamp_confocal')
        for key in required:
            if key not in f:raise DataReview('UNSUPPORTED_HDF5_LAYOUT')
        shape=f['gcamp/trace_array_original'].shape
        if len(shape)!=2 or np.prod(shape)>MAX_CELLS:raise DataReview('HDF5_ARRAY_OVER_CAP')
        t=f['timing/timestamp_confocal'][()]
        a=f['gcamp/trace_array_original'][()]
        # Julia HDF5 writers may reverse on-disk dimension ordering.
        if a.shape[0]==len(t) and a.shape[1]!=len(t):a=a.T
        elif a.shape[1]!=len(t) or a.shape[0]==a.shape[1]:raise DataReview('HDF5_ORIENTATION_AMBIGUOUS')
        uid=source_name[:-len('-cudata.h5')]
        md={'uid':uid,'paper_id':'dryad_w9ghx3g4v','n_neuron':a.shape[0],'dataset_type':['copper_boundary','freely_moving'],
            'source_filename':source_name}
        obj={'metadata':md,'gcamp':{'trace_array_original':a},'timing':{'timestamp_confocal':t},'behavior':{}}
        if 'gcamp/trace_array' in f:
            z=f['gcamp/trace_array'][()]
            if z.shape!=a.shape and z.T.shape==a.shape:z=z.T
            obj['gcamp']['trace_array']=z
        for key in BEHAVIOR_KEYS:
            if 'behavior/'+key in f:obj['behavior'][key]=f['behavior/'+key][()]
        data=prepare_www(obj)
        data['metadata'].update(source_format='dryad-copper-hdf5',animal_id=uid,animal_identity_status='source_verified_within_dataset',
            source_group_id='dryad:w9ghx3g4v:animal:'+uid,
            identity_note='Named neuron mapping is in NeuroPAL_labels.jld2; pending verified adapter. ROI columns are not named neuron IDs.',
            time_unit_evidence='Author README: timing/timestamp_confocal, approximately 1.67 Hz; actual timestamps preserved.')
        for c in data['metadata']['behavior_channels']:
            if c['name']=='velocity':c['source_unit']='mm/s'
            if c['name']=='head_angle':c['sign_convention']='source orientation; may differ by animal'
        return data

def write_record(data,destination,source=None):
    root=pathlib.Path(destination);root.mkdir(parents=True,exist_ok=True)
    a=data['signal'];t=data['time_source'];sigsha=sha_array(a);tsha=sha_array(t)
    key=hashlib.sha256((sigsha+tsha).encode()).hexdigest()[:32]
    target=root/key
    if (target/'metadata.json').is_file():
        previous=json.loads((target/'metadata.json').read_text())
        if previous['signal_sha256']!=sigsha or previous['timestamps_sha256']!=tsha:raise DataReview('RECORD_HASH_CONFLICT')
        return previous,0
    temp=root/('.building-'+uuid.uuid4().hex);temp.mkdir()
    try:
        np.save(temp/'signal.npy',a,allow_pickle=False)
        np.save(temp/'valid.npy',data['valid'],allow_pickle=False)
        np.save(temp/'time_s.npy',data['time_s'],allow_pickle=False)
        np.save(temp/'time_source.npy',t,allow_pickle=False)
        if data['author_zscore'] is not None:np.save(temp/'signal_author_zscore.npy',data['author_zscore'],allow_pickle=False)
        if data['behavior'] is not None:
            np.save(temp/'behavior.npy',data['behavior'],allow_pickle=False)
            np.save(temp/'behavior_valid.npy',np.isfinite(data['behavior']),allow_pickle=False)
        n_finite=int(data['valid'].sum());md=data['metadata']
        eligible=md['source_original_available'] and not md['control'] and any(not n['is_constant'] and n['n_finite']>=40 for n in data['neurons'])
        qc='control' if md['control'] else ('usable' if eligible and data['valid'].all() else ('partial' if eligible else 'review'))
        manifest={**md,'record_key':key,'parser_version':PARSER_VERSION,'signal_sha256':sigsha,'timestamps_sha256':tsha,
            'n_timepoints':a.shape[0],'n_neurons':a.shape[1],'n_named_neurons':sum(bool(n['neuron_name']) for n in data['neurons']),
            'finite_fraction':n_finite/a.size,'median_dt_s':data['median_dt_s'],'n_segments':len(data['segments']),
            'axis_order':['time','neuron'],'dtype':'float32','mask_definition':'finite numerical values; not a complete biological quality mask',
            'qc_state':qc,'training_eligible':bool(eligible),'evaluation_scope':'chronological within-record forecasting; cross-animal grouping requires identity review',
            'source':source or {}}
        jwrite(temp/'metadata.json',clean(manifest));jwrite(temp/'neurons.json',clean(data['neurons']))
        jwrite(temp/'segments.json',{'intervals':data['segments'],'indexing':'zero_based_half_open','gap_threshold':'1.5 * median timestamp interval'})
        jwrite(temp/'events.json',{'source_index':md['events_source_index'],'index_origin':md['event_index_origin'],'converted_to_seconds':False})
        jwrite(temp/'behavior_channels.json',md['behavior_channels'])
        nbytes=sum(p.stat().st_size for p in temp.iterdir())
        os.replace(temp,target)
        return manifest,nbytes
    finally:
        if temp.exists() and temp.resolve().parent==root.resolve() and temp.name.startswith('.building-'):shutil.rmtree(temp)

def register(agent,file_id,member,manifest):
    m=manifest;k=m['record_key'];rel='neural/records/'+k
    row=(k,PARSER_VERSION,m['source_format'],m['source_record_id'],m['source_group_id'],m['animal_identity_status'],m['signal_kind'],
        m['n_timepoints'],m['n_neurons'],m['n_named_neurons'],m['finite_fraction'],m['n_segments'],m['median_dt_s'],m['qc_state'],
        int(m['training_eligible']),m['signal_sha256'],m['timestamps_sha256'],rel,json.dumps(m,ensure_ascii=False),agent.now())
    agent.run('INSERT OR IGNORE INTO neural_recordings VALUES ('+','.join('?' for _ in row)+')',row)
    did=agent.rows('SELECT dataset_id FROM dataset_files WHERE file_id=? ORDER BY dataset_id LIMIT 1',(file_id,))
    agent.run('INSERT OR IGNORE INTO neural_record_sources VALUES (?,?,?,?,?)',(k,file_id,member,did[0]['dataset_id'] if did else None,m['source_record_id']))
    neurons=json.loads((agent.root/rel/'neurons.json').read_text())
    with agent.lock:
        agent.db.executemany('INSERT OR IGNORE INTO neural_neurons VALUES (?,?,?,?,?,?,?,?,?)',[
            (k,n['column_index'],n['source_row_index'],n['neuron_name'],n['neuron_class'],n['confidence_source'],n['n_finite'],int(n['is_constant']),json.dumps(n,ensure_ascii=False)) for n in neurons])
        agent.db.commit()

def parse_file(agent,job):
    fid=int(job['target_id']);f=agent.rows('SELECT * FROM files WHERE file_id=?',(fid,))[0]
    if f['file_state']!='verified':raise DataReview('SOURCE_NOT_VERIFIED')
    path=agent.root/f['object_relpath'];name=f['original_name'];parsed=0;reviewed=0
    def consume(data,member):
        nonlocal parsed
        if agent.stopping():raise InterruptedError('STOPPING')
        estimate=int(data['signal'].size*16+len(data['time_s'])*64+4_000_000)
        agent.reserve(estimate);written=0
        try:
            md,written=write_record(data,agent.root/'neural/records',{'file_id':fid,'source_sha256':f['sha256'],'member':member})
            register(agent,fid,member,md);parsed+=1
            agent.heartbeat(job)
            agent.log('neural_record_standardized',record_key=md['record_key'],source_uid=md['source_record_id'],timepoints=md['n_timepoints'],neurons=md['n_neurons'],named_neurons=md['n_named_neurons'],qc=md['qc_state'])
        finally:agent.release(estimate,written)
    if name.lower().endswith('.tar.bz2'):
        total=0
        with bz2.open(path,'rb') as stream,tarfile.open(fileobj=stream,mode='r|') as archive:
            for ix,member in enumerate(archive):
                if ix>=10000:raise DataReview('ARCHIVE_MEMBER_LIMIT')
                total+=member.size
                if total>2_000_000_000:raise DataReview('ARCHIVE_EXPANDED_LIMIT')
                if not member.isfile() or not member.name.lower().endswith('.json'):continue
                if member.size>MAX_JSON:
                    reviewed+=1;agent.log('neural_member_review',file_id=fid,member=member.name,reason='JSON_MEMBER_OVER_CAP');continue
                if agent.rows('SELECT 1 FROM neural_record_sources WHERE file_id=? AND source_member=?',(fid,member.name)):continue
                try:
                    with archive.extractfile(member) as stream_member:obj=json.load(stream_member)
                    data=prepare_www(obj);del obj
                    consume(data,member.name);del data
                except (DataReview,KeyError,ValueError,TypeError) as e:
                    reviewed+=1;agent.log('neural_member_review',file_id=fid,member=member.name,reason=str(e)[:180])
    elif name.lower().endswith(('.json','.json.bz2')):
        with (bz2.open(path,'rb') if name.endswith('.bz2') else path.open('rb')) as stream:
            raw=stream.read(MAX_JSON+1)
        if len(raw)>MAX_JSON:raise DataReview('JSON_MEMBER_OVER_CAP')
        obj=json.loads(raw);del raw
        data=prepare_www(obj);del obj;consume(data,name)
    elif name.endswith('-cudata.h5'):
        consume(prepare_copper_h5(path,name),name)
    else:raise DataReview('NEURAL_ADAPTER_PENDING_RAW_PRESERVED')
    existing=agent.rows('SELECT count(*) n FROM neural_record_sources WHERE file_id=?',(fid,))[0]['n']
    if not parsed and not existing:raise DataReview('NO_SUPPORTED_NEURAL_RECORDS')
    agent.log('neural_file_parsed',file_id=fid,new_records=parsed,linked_records=existing,reviewed_members=reviewed)

