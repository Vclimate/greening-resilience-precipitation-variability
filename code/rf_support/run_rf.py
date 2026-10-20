"""Original-source RF/TPE/OOB experiment, no rejected inputs or fitted caches."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import h5py
import joblib
import numpy as np
import pandas as pd
import optuna
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT.parent
SOURCE = ROOT / 'data/analysis.parquet'
SEED = 20260924
CONFIG = dict(version='original_source_drop_P_Ta_rf_v1', seed=SEED, trees=300,
    threads_per_model=14, trials=24, startup_trials=8, outer_test_fraction=.3,
    inner_validation_fraction=.25, training_cap=80000, validation_cap=30000,
    min_samples_leaf=[1,64], max_features=[2,20], removed=['P','Ta'],
    bootstrap=True, max_samples=None, criterion='gini', permutation_repeats=3,
    threshold=.5, objective='inner validation accuracy; full-inner shortlist verification',
    sample='original mask, original OLS P<=.05, complete 22 original predictors',
    outliers='training-only response 1.5 IQR; held-out samples unchanged',
    validation='random stratified 70/30; not spatially independent validation',
    matlab_replication=False)
REFERENCE = dict(min_samples_leaf=5, max_features=4)


def log(s):
    print(time.strftime('%H:%M:%S')+' '+s, flush=True)


def dump(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def mat(path, key, allowed):
    path=Path(path).resolve()
    if path not in allowed:
        raise ValueError(f'Input not explicitly approved: {path}')
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as f:
            a=f[key][:].T
    else:
        a=loadmat(path,variable_names=[key])[key]
    assert a.shape==(3600,7200)
    return a


def prepare():
    policy=read(APPROVED/'approved_sources.json')
    origin=Path(policy['approved_root']).resolve()
    forbidden=Path(policy['forbidden_root']).resolve()
    specs=policy['predictors']
    candidates=[s['variable'] for s in specs]
    columns=[s for s in candidates if s not in CONFIG['removed']]
    assert len(candidates)==22 and len(columns)==20
    yp=Path(policy['response']['path']).resolve()
    sp=Path(policy['significance_layer']['path']).resolve()
    lcpath=origin/'input/005IBGP_SLC.mat'
    mdnpath=origin/'input/Variables/kndvi16d/kndvi_MODIS_orig_acf_wh_map_sw005.mat'
    maskpath=origin.parent/'MASK_IGBP不变_MDN缺失值05.mat'
    allowed={Path(s['path']).resolve() for s in specs}|{yp,sp,lcpath,mdnpath,maskpath}
    assert all(not p.is_relative_to(forbidden) and p.is_file() for p in allowed)
    assert all(p.is_relative_to(origin) or p==maskpath for p in allowed)
    hashes={str(p):digest(p) for p in sorted(allowed)}
    manifest=dict(configuration=CONFIG, source_files_sha256=hashes,
        approved_root=str(origin), selected_features=columns, candidates=candidates,
        source_policy_sha256=digest(APPROVED/'approved_sources.json'),
        forbidden_data_read=False, prepared_table_from_other_analysis_used=False)
    if SOURCE.exists():
        prior=read(ROOT/'data/source_manifest.json')
        assert prior==manifest and digest(SOURCE)==read(ROOT/'data/qa.json')['table_sha256']
        log('Verified existing original-source table and all source checksums')
        return
    log('Reading original RF sources only')
    y=mat(yp,'Xtrend',allowed)
    p=mat(yp,'Xpval',allowed)
    lc=mat(lcpath,'IBGP_SLC',allowed)
    mdn=mat(mdnpath,'MDN',allowed)
    stable=mat(maskpath,'MASK',allowed)
    base=(stable==1)&np.isfinite(lc)&(lc!=0)&(mdn<.5)&np.isfinite(y)
    significant=base&np.isfinite(p)&(p<=.05)
    saved=mat(sp,policy['significance_layer']['dataset'],allowed)
    assert np.array_equal(significant[base],np.isfinite(saved[base]))
    np.testing.assert_array_equal(y[significant],saved[significant])
    ids=np.flatnonzero(significant)
    row,col=np.divmod(ids,7200)
    df=pd.DataFrame(dict(pixel_id=ids.astype('int32'),longitude=-180+(col+.5)*.05,
        latitude=90-(row+.5)*.05,vegetation=np.where(lc.ravel()[ids]<=5,'Forest','Non-forest'),
        y_continuous=y.ravel()[ids],y=(y.ravel()[ids]>=0).astype('int8'),tac_pvalue=p.ravel()[ids]))
    del y,p,lc,mdn,stable,base,significant,saved
    audits=[]
    for s in specs:
        a=mat(s['path'],s['dataset'],allowed).ravel()[ids].astype('float32')
        invalid=(~np.isfinite(a))|(np.abs(a)>1e30)|np.isin(a,[-9999.,-32768.])
        a[invalid]=np.nan
        df[s['variable']]=a
        audits.append(dict(**s,invalid_significant_pixels=int(invalid.sum())))
        log(f"Original {s['variable']}: invalid {invalid.sum():,}")
    df=df.loc[np.isfinite(df[candidates]).all(axis=1)].reset_index(drop=True)
    assert df.pixel_id.is_unique
    counts=df.groupby('vegetation').size().to_dict()
    assert counts=={'Forest':234144,'Non-forest':502189},counts
    # Compare only pixel membership against the just-completed original-source VIF check.
    expected=pd.read_parquet(APPROVED/'vif_original_p005/vif_sample_membership.parquet')
    expected=expected.loc[expected.scope.eq('original_complete22')]
    assert np.array_equal(np.sort(expected.pixel_id),np.sort(df.pixel_id))
    SOURCE.parent.mkdir(exist_ok=True)
    df.to_parquet(SOURCE,index=False)
    pd.DataFrame(audits).to_csv(ROOT/'data/input_dictionary.csv',index=False,encoding='utf-8-sig')
    dump(ROOT/'data/source_manifest.json',manifest)
    dump(ROOT/'data/qa.json',dict(n=len(df),counts=counts,source_hashes_verified=True,
        table_sha256=digest(SOURCE),p_exactly_005=int(df.tac_pvalue.eq(.05).sum()),
        zero_trend=int(df.y_continuous.eq(0).sum()),matches_original_source_vif_population=True))
    log(f'Prepared {counts}')


def iqr(df,idx):
    a=df.y_continuous.to_numpy()[idx]
    q1,q3=np.quantile(a,[.25,.75])
    low,high=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
    good=(a>=low)&(a<=high)
    return idx[good],dict(low=float(low),high=float(high),removed=int((~good).sum()))


def sample(idx,y,cap,seed):
    if len(idx)<=cap:return idx
    selected,_=train_test_split(idx,train_size=cap,stratify=y[idx],random_state=seed)
    return np.sort(selected)


def rf(params):
    return RandomForestClassifier(n_estimators=CONFIG['trees'],random_state=SEED,
        n_jobs=CONFIG['threads_per_model'],bootstrap=True,max_samples=None,criterion='gini',
        max_depth=None,class_weight=None,min_samples_split=2,**params)


def metrics(y,p,baseline):
    pred=(p>=.5).astype('int8')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),roc_auc=float(roc_auc_score(y,p)),
        majority_baseline_accuracy=float(accuracy_score(y,baseline)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n_test=len(y),
        recall_declining=float(tp/(tp+fn)),recall_increasing=float(tn/(tn+fp)))


def train(veg):
    folder=ROOT/'models'/veg
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training_complete.json').exists():
        c=read(folder/'training_complete.json')
        assert c['configuration']==CONFIG and c['model_sha256']==digest(folder/'model.joblib')
        return
    protocol=dict(configuration=CONFIG,table_sha256=digest(SOURCE),
        code_sha256=digest(__file__),versions={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','optuna','pandas']})
    if (folder/'protocol.json').exists():assert read(folder/'protocol.json')==protocol
    else:dump(folder/'protocol.json',protocol)
    names=read(ROOT/'data/source_manifest.json')['selected_features']
    df=pd.read_parquet(SOURCE)
    df=df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
    X,y=df[names].to_numpy('float32'),df.y.to_numpy()
    outer,test=train_test_split(np.arange(len(df)),test_size=.3,stratify=y,random_state=SEED)
    outer,test=np.sort(outer),np.sort(test)
    fit,outer_iqr=iqr(df,outer)
    inner0,val=train_test_split(outer,test_size=.25,stratify=y[outer],random_state=SEED+1100)
    inner0,val=np.sort(inner0),np.sort(val)
    inner,inner_iqr=iqr(df,inner0)
    fs=sample(inner,y,CONFIG['training_cap'],SEED+1101)
    vs=sample(val,y,CONFIG['validation_cap'],SEED+1102)
    assert not np.intersect1d(outer,test).size and not np.intersect1d(inner0,val).size
    assert np.array_equal(np.sort(np.r_[outer,test]),np.arange(len(df)))
    assert np.array_equal(np.sort(np.r_[inner0,val]),outer)
    if CONFIG.get('comparison_split_root'):
        old=np.load(Path(CONFIG['comparison_split_root'])/veg/'split_indices.npz')
        for key,value in dict(training_before_iqr=outer,training=fit,test=test,
            inner_before_iqr=inner0,inner_training=inner,inner_validation=val,
            sampled_inner_training=fs,sampled_inner_validation=vs).items():
            np.testing.assert_array_equal(value,old[key])
    vif=[]
    for scope,indices in [('outer_training',fit),('inner_training',inner)]:
        C=np.corrcoef(X[indices].astype('float64'),rowvar=False)
        values=np.diag(np.linalg.inv(C))
        assert np.isfinite(values).all() and values.max()<=10,(scope,values.max())
        vif.extend(dict(scope=scope,variable=n,vif=float(v)) for n,v in zip(names,values))
    pd.DataFrame(vif).to_csv(folder/'training_vif.csv',index=False)
    np.savez_compressed(folder/'split_indices.npz',training_before_iqr=outer,training=fit,
        test=test,inner_before_iqr=inner0,inner_training=inner,inner_validation=val,
        sampled_inner_training=fs,sampled_inner_validation=vs)
    dump(folder/'design.json',dict(n_eligible=len(df),n_training_before_iqr=len(outer),n_training=len(fit),
        n_test=len(test),n_inner_training=len(inner),n_inner_validation=len(val),
        selected_features=names,removed_features=CONFIG['removed'],outer_iqr=outer_iqr,inner_iqr=inner_iqr,
        test_not_used_for_tuning=True,training_vif_max=max(r['vif'] for r in vif),
        population='Original-source complete-22 significant pixels, NOT legacy 76-column/IQR population',
        feature_choice=CONFIG.get('feature_choice','User-fixed P/Ta deletion following full-population descriptive VIF; confirmed on training splits'),
        matlab_differences='CART Gini/with-replacement bootstrap vs legacy curvature/10% without replacement; train-only Y-IQR vs legacy joint-IQR; original sample filters differ'))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_path=folder/'sampler.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=SEED,n_startup_trials=8)
    storage=optuna.storages.RDBStorage(url='sqlite:///'+(folder/'tuning.sqlite3').as_posix())
    study=optuna.create_study(study_name=CONFIG['version'],direction='maximize',sampler=sampler,storage=storage,load_if_exists=True)
    assert study.user_attrs.get('configuration') in (None,CONFIG)
    study.set_user_attr('configuration',CONFIG)
    for t in study.trials:
        if t.state==optuna.trial.TrialState.RUNNING:study.tell(t.number,state=optuna.trial.TrialState.FAIL)
    if not study.trials:study.enqueue_trial(REFERENCE)
    def objective(trial):
        params=dict(min_samples_leaf=trial.suggest_int('min_samples_leaf',1,64,log=True),
            max_features=trial.suggest_int('max_features',2,min(CONFIG['max_features'][1],len(names))))
        duplicate=next((t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.params==params),None)
        if duplicate is not None:
            trial.set_user_attr('auc',duplicate.user_attrs['auc'])
            trial.set_user_attr('duplicate_of',duplicate.number)
            return duplicate.value
        model=rf(params).fit(X[fs],y[fs])
        p=model.predict_proba(X[vs])[:,1]
        trial.set_user_attr('auc',float(roc_auc_score(y[vs],p)))
        return float(accuracy_score(y[vs],p>=.5))
    def progress(study,trial):
        study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
        joblib.dump(study.sampler,sampler_path)
        log(f'{veg}: TPE {trial.number+1}/24; inner accuracy {trial.value:.2%}; {trial.params}')
    log(f'{veg}: {len(fit):,} training, {len(test):,} test; starting TPE')
    done=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,24-done),callbacks=[progress])
    ranked=sorted([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE],key=lambda t:(-t.value,-t.user_attrs['auc'],t.number))
    choices=[]
    seen=set()
    for t in ranked:
        key=tuple(sorted(t.params.items()))
        if key not in seen:choices.append((f'trial_{t.number}',t.params));seen.add(key)
        if len(choices)==3:break
    if tuple(sorted(REFERENCE.items())) not in seen:choices.append(('reference',REFERENCE))
    scores=[]
    for name,params in choices:
        cp=folder/f'inner_{name}.json'
        if cp.exists():entry=read(cp)
        else:
            model=rf(params).fit(X[inner],y[inner])
            p=model.predict_proba(X[val])[:,1]
            entry=dict(name=name,params=params,accuracy=float(accuracy_score(y[val],p>=.5)),auc=float(roc_auc_score(y[val],p)))
            dump(cp,entry)
        scores.append(entry)
        log(f'{veg}: full inner {name}: {entry["accuracy"]:.2%}')
    chosen=sorted(scores,key=lambda s:(-s['accuracy'],-s['auc'],s['name']))[0]
    dump(folder/'selection.json',dict(chosen=chosen,candidates=scores,test_used=False))
    storage.remove_session();storage.engine.dispose()
    model=rf(chosen['params']).fit(X[fit],y[fit])
    p=model.predict_proba(X[test])[:,1]
    baseline=np.full(len(test),int(y[fit].mean()>=.5),dtype='int8')
    result=metrics(y[test],p,baseline)
    pred=df.iloc[test][['pixel_id','longitude','latitude','y','tac_pvalue']].copy()
    pred['probability'],pred['prediction'],pred['baseline']=p,(p>=.5).astype('int8'),baseline
    pred.to_parquet(folder/'predictions.parquet',index=False)
    dump(folder/'metrics.json',result)
    joblib.dump(model,folder/'model.joblib',compress=3)
    restored=joblib.load(folder/'model.joblib')
    np.testing.assert_array_equal(restored.predict_proba(X[test[:100]])[:,1],p[:100])
    dump(folder/'training_complete.json',dict(configuration=CONFIG,model_sha256=digest(folder/'model.joblib')))
    log(f'{veg}: FINAL accuracy={result["accuracy"]:.3%}, AUC={result["roc_auc"]:.4f}; {chosen["params"]}')


def one_tree(tree,k,X,y,target=None):
    if target is not None and target.exists():
        with np.load(target) as a:return a['delta'],float(a['baseline_error']),int(a['n_oob'])
    oob=_generate_unsampled_indices(tree.random_state,len(y),len(y))
    values=np.array(X[oob],dtype='float32',order='C')
    labels=y[oob]
    base=float(np.mean(tree.predict(values,check_input=False)!=labels))
    delta=np.zeros((3,X.shape[1]))
    for j in range(X.shape[1]):
        original=values[:,j].copy()
        for r in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,1200,k,j,r]))
            values[:,j]=original[rng.permutation(len(original))]
            delta[r,j]=np.mean(tree.predict(values,check_input=False)!=labels)-base
        values[:,j]=original
    if target is not None:np.savez_compressed(target,delta=delta,baseline_error=base,n_oob=len(oob))
    return delta,base,len(oob)


def importance(veg):
    folder=ROOT/'models'/veg
    output=ROOT/'results'/veg
    output.mkdir(parents=True,exist_ok=True)
    names=read(folder/'design.json')['selected_features']
    modelhash=digest(folder/'model.joblib')
    if (output/'complete.json').exists():
        assert read(output/'complete.json')['model_sha256']==modelhash
        return
    df=pd.read_parquet(SOURCE)
    df=df.loc[df.vegetation.eq(veg)].reset_index(drop=True)
    idx=np.load(folder/'split_indices.npz')['training']
    X,y=df[names].to_numpy('float32')[idx],df.y.to_numpy()[idx]
    model=joblib.load(folder/'model.joblib')
    assert model.bootstrap and model.max_samples is None and np.array_equal(model.classes_,[0,1])
    manifest=dict(configuration=CONFIG,model_sha256=modelhash,table_sha256=digest(SOURCE),
        training_index_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),features=names)
    if (output/'manifest.json').exists():assert read(output/'manifest.json')==manifest
    else:dump(output/'manifest.json',manifest)
    chunks=output/'trees'
    chunks.mkdir(exist_ok=True)
    log(f'{veg}: OOB permutation, 300 trees x {len(names)} variables x 3 repeats')
    stream=joblib.Parallel(n_jobs=14,prefer='threads',return_as='generator')(
        joblib.delayed(one_tree)(t,k,X,y,chunks/f'{k:03d}.npz') for k,t in enumerate(model.estimators_))
    results=[]
    for k,result in enumerate(stream,1):
        results.append(result)
        if k%25==0:log(f'{veg}: OOB {k}/300 trees')
    delta=np.stack([r[0] for r in results])
    raw=delta.mean(axis=0)
    sd=delta.std(axis=0,ddof=1)
    assert np.isfinite(delta).all() and (sd>0).all()
    normalized=raw/sd
    table=pd.DataFrame(dict(variable=names,raw_delta_error=raw.mean(axis=0),
        raw_delta_error_pp=100*raw.mean(axis=0),standardized_importance=normalized.mean(axis=0),
        standardized_repeat_sd=normalized.std(axis=0,ddof=1),raw_repeat_sd_pp=100*raw.std(axis=0,ddof=1)))
    table['rank_standardized']=table.standardized_importance.rank(ascending=False,method='min').astype(int)
    table['rank_raw']=table.raw_delta_error.rank(ascending=False,method='min').astype(int)
    table=table.sort_values('rank_standardized')
    table.to_csv(output/'importance.csv',index=False)
    np.savez_compressed(output/'per_tree_deltas.npz',delta=delta,features=names,
        baseline_error=[r[1] for r in results],n_oob=[r[2] for r in results])
    dump(output/'complete.json',dict(**manifest,n_trees=len(results),
        definition='Mean single-tree OOB error increase divided by sample SD across trees, calculated for each repeat then averaged',
        caveat='Not impurity importance, not ensemble error loss, not probability change; permutation-repeat SD is Monte Carlo variation, not a confidence interval'))
    log(f'{veg}: OOB ranking: '+', '.join(table.variable.head(6)))


def self_test():
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=400,n_features=5,n_informative=3,random_state=7)
    X=X.astype('float32')
    m=RandomForestClassifier(n_estimators=80,oob_score=True,random_state=7).fit(X,y)
    total=np.zeros((len(y),2));count=np.zeros(len(y))
    for t in m.estimators_:
        oob=_generate_unsampled_indices(t.random_state,len(y),len(y))
        total[oob]+=t.predict_proba(X[oob]);count[oob]+=1
    np.testing.assert_allclose(total/count[:,None],m.oob_decision_function_,atol=1e-14)
    a=one_tree(m.estimators_[0],0,X,y)
    b=one_tree(m.estimators_[0],0,X,y)
    np.testing.assert_array_equal(a[0],b[0])
    oob=_generate_unsampled_indices(m.estimators_[0].random_state,len(y),len(y))
    xx=X[oob].copy()
    rng=np.random.default_rng(np.random.SeedSequence([SEED,1200,0,0,0]))
    xx[:,0]=xx[rng.permutation(len(xx)),0]
    manual=np.mean(m.estimators_[0].predict(xx)!=y[oob])-np.mean(m.estimators_[0].predict(X[oob])!=y[oob])
    assert np.isclose(a[0][0,0],manual)
    dump(ROOT/'unit_checks.json',dict(reconstructed_oob_matches_sklearn=True,reproducible=True,manual_permutation_matches=True))


def workflow(worker_script=None):
    prepare()
    self_test()
    handles=[]
    try:
        for veg in ['Forest','Non-forest']:
            logs=ROOT/'logs'
            logs.mkdir(exist_ok=True)
            path=logs/f'{veg}.log'
            f=path.open('w',encoding='utf-8')
            script=Path(worker_script).resolve() if worker_script else Path(__file__).resolve()
            p=subprocess.Popen([sys.executable,'-u',str(script),'--stage','model','--vegetation',veg],
                stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            handles.append(dict(vegetation=veg,process=p,file=f,path=path,offset=0))
        while True:
            for h in handles:
                with h['path'].open('r',encoding='utf-8',errors='replace') as f:
                    f.seek(h['offset']);new=f.read();h['offset']=f.tell()
                if new:print(new,end='',flush=True)
            if all(h['process'].poll() is not None for h in handles):break
            time.sleep(5)
        for h in handles:
            if h['process'].returncode!=0:raise RuntimeError(h['vegetation']+' failed; inspect log')
    finally:
        for h in handles:
            if h['process'].poll() is None:
                h['process'].terminate();h['process'].wait()
            h['file'].close()
    log('Both RF models and OOB importance completed.')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','prepare','model','test'],default='all')
    parser.add_argument('--vegetation',choices=['Forest','Non-forest'])
    args=parser.parse_args()
    if args.stage=='all':workflow()
    elif args.stage=='prepare':prepare()
    elif args.stage=='test':self_test()
    else:
        assert args.vegetation
        train(args.vegetation)
        importance(args.vegetation)
