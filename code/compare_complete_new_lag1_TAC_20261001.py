"""Audit 21 new lag1 exports against the old exports and the saved RF response."""
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DLL=os.add_dll_directory(str(Path(sys.base_prefix)/'Library/bin'))
import pyarrow.parquet
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from scipy import stats
import statsmodels.api as sm
from audit_GEE_exports_pre_RF import ROOT,DEST,digest
from compare_new_lag1_TAC_20260928 import readmat,summarize

NEW=Path('C:/Users/user/Downloads/KNDVI_TAC_lag1_5year_window_from_MOD13C1_005deg')
OLD=ROOT/'input/KNDVI_AC1_5year_window-01-01_005'
OUT=ROOT/'output/TAC_lag1_complete_new_vs_original_20261001'
YEARS=np.arange(2005,2026)
GRID=from_origin(-180,80,.05,.05)


def fit_pixels(a,years):
    a=a.astype('float64')
    good=np.isfinite(a)
    n=good.sum(axis=0)
    x=(years-years.mean()).astype(float)
    sx=x@good; sxx=(x*x)@good
    z=np.where(good,a,0)
    sy=z.sum(axis=0);syy=(z*z).sum(axis=0);sxy=x@z
    with np.errstate(divide='ignore',invalid='ignore'):
        xx=sxx-sx*sx/n;xy=sxy-sx*sy/n;yy=syy-sy*sy/n
        slope=xy/xx
        sse=np.maximum(0,yy-slope*xy)
        se=np.sqrt(sse/(n-2)/xx)
        p=2*stats.t.sf(np.abs(slope/se),n-2)
    bad=(n<=2)|~np.isfinite(slope)|~np.isfinite(p)
    slope[bad]=np.nan;p[bad]=np.nan
    return slope,p,n


def write_map(name,ids,values):
    a=np.full((2800,7200),-9999,dtype='float32')
    a.ravel()[ids]=np.where(np.isfinite(values),values,-9999)
    with rasterio.open(OUT/name,'w',driver='GTiff',height=2800,width=7200,count=1,
        dtype='float32',crs='EPSG:4326',transform=GRID,nodata=-9999,compress='deflate',tiled=True) as f:
        f.write(a,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=json.loads((DEST/'RF_input_manifest_22.json').read_text(encoding='utf-8'))
    support={s['variable']:s for s in m['response_and_masks']}
    lc=readmat(support['land_cover']);stable=readmat(support['stable_vegetation_mask'])
    mdn=readmat(support['MDN']);saved_y=readmat(support['delta_TAC']);saved_p=readmat(support['TAC_pvalue'])
    base=(stable==1)&np.isfinite(lc)&(lc!=0)&np.isfinite(mdn)&(mdn<.5)&np.isfinite(saved_y)
    ids=np.flatnonzero(base);rows,cols=np.divmod(ids,7200)
    lc=lc.ravel()[ids]
    sy=saved_y.ravel()[ids];sp=saved_p.ravel()[ids]
    del stable,mdn,saved_y,saved_p,base
    weights=np.cos(np.deg2rad(80-(rows+.5)*.05))
    scopes={'Global':np.ones(len(ids),bool),'Forest':lc<=5,'Non-forest':lc>5}
    stacks={v:np.empty((21,len(ids)),dtype='float32') for v in ['old','new']}
    files=[]
    expected={f'KNDVI_TAC_lag1_5year_window_{year}_01_01.tif' for year in YEARS}
    assert {p.name for p in NEW.glob('*.tif')}==expected
    tasks=[(v,int(y),OLD/f'KNDVI_AC1_5year_window_{y}_01_01.tif' if v=='old' else NEW/f'KNDVI_TAC_lag1_5year_window_{y}_01_01.tif') for v in stacks for y in YEARS]
    def worker(t):
        v,y,p=t
        with rasterio.open(p) as f:
            assert f.crs.to_epsg()==4326 and f.transform.almost_equals(GRID)
            assert f.shape in [(2800,7200),(2801,7201)]
            raw=f.read(1,masked=True,window=Window(0,0,7200,2800)).astype('float32').filled(np.nan)
            raw[(~np.isfinite(raw))|(np.abs(raw)>1e30)|np.isin(raw,[-9999,-32768])]=np.nan
            vals=raw.ravel()[ids]
            stacks[v][y-2005]=vals
            record=dict(version=v,file_year=y,reference_year=y-1,path=str(p),sha256=digest(p),
                shape=list(f.shape),band=f.descriptions[0],nodata=f.nodata,
                global_valid=int(np.isfinite(raw).sum()),global_abs_gt_1=int((np.abs(raw)>1).sum()),
                global_min=float(np.nanmin(raw)),global_max=float(np.nanmax(raw)),
                base_valid=int(np.isfinite(vals).sum()),base_abs_gt_1=int((np.abs(vals)>1).sum()))
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(worker,tasks),1):
            files.append(r)
            if i%7==0:print(f'Read {i}/42 rasters',flush=True)
    (OUT/'input_inventory.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
    annual=[]
    for i,year in enumerate(YEARS):
        a,b=stacks['old'][i],stacks['new'][i]
        for scope,mask in scopes.items():
            oldvalid=mask&np.isfinite(a);newvalid=mask&np.isfinite(b)
            for rule in ['all_finite','both_in_minus1_plus1']:
                both=oldvalid&newvalid
                if rule!='all_finite':both&=(np.abs(a)<=1)&(np.abs(b)<=1)
                annual.append(dict(file_year=int(year),scope=scope,rule=rule,
                    lost=int((oldvalid&~newvalid).sum()),gained=int((newvalid&~oldvalid).sum()),
                    **summarize(a[both].astype(float),b[both].astype(float))))
    annual=pd.DataFrame(annual)
    annual.to_csv(OUT/'annual_comparison.csv',index=False,encoding='utf-8-sig')
    print('Annual comparison completed',flush=True)

    pixel_rows=[];regional=[];trajectories=[];reproduction=None
    rf_labels=None
    for period,nperiod in [('file2005_2023_original_RF',19),('file2005_2024_lag_common',20),('file2005_2025_extension',21)]:
        for rule in ['minus1_plus1','zero_plus1_RF_rule']:
            filtered={}
            slope={};pvalue={}
            for v in stacks:
                a=stacks[v][:nperiod].copy()
                a[(a>1)|(a<(-1 if rule=='minus1_plus1' else 0))]=np.nan
                filtered[v]=a
                slope[v],pvalue[v],_=fit_pixels(a,YEARS[:nperiod])
            paired=np.isfinite(slope['old'])&np.isfinite(slope['new'])
            if nperiod==19 and rule=='zero_plus1_RF_rule':
                compare=np.isfinite(slope['old'])&np.isfinite(sy)
                reproduction=dict(paired=int(compare.sum()),max_abs_slope_error=float(np.max(np.abs(slope['old'][compare]-sy[compare]))),
                    pearson_r=float(np.corrcoef(slope['old'][compare],sy[compare])[0,1]),
                    old_sig_mismatch=int(((pvalue['old'][compare]<.05)!=(sp[compare]<.05)).sum()),
                    pvalue_max_error=float(np.max(np.abs(pvalue['old'][compare]-sp[compare]))))
                print('Original RF response reproduction',json.dumps(reproduction),flush=True)
                label=pd.DataFrame(dict(pixel_id=((rows+200)*7200+cols).astype('int32'),
                    vegetation=np.where(lc<=5,'Forest','Non-forest'),saved_old_slope=sy,saved_old_p=sp,
                    old_recomputed_slope=slope['old'],new_slope=slope['new'],new_p=pvalue['new']))
                label.to_parquet(OUT/'RF_response_comparison_original_period.parquet',index=False)
                rf_labels=label
            for scope,mask in scopes.items():
                keep=paired&mask
                a,b=slope['old'][keep],slope['new'][keep]
                oldsig=pvalue['old'][keep]<.05;newsig=pvalue['new'][keep]<.05
                row=dict(period=period,rule=rule,scope=scope,paired=int(keep.sum()),
                    slope_pearson_r=float(np.corrcoef(a,b)[0,1]),slope_MAE=float(np.mean(np.abs(b-a))),
                    slope_sign_agreement_pct=float(100*np.mean((a>=0)==(b>=0))),
                    old_positive_area_pct=float(100*np.average(a>=0,weights=weights[keep])),
                    new_positive_area_pct=float(100*np.average(b>=0,weights=weights[keep])),
                    old_significant=int(oldsig.sum()),new_significant=int(newsig.sum()),
                    old_significant_retained=int((oldsig&newsig).sum()),
                    old_significant_lost=int((oldsig&~newsig).sum()),new_only_significant=int((~oldsig&newsig).sum()),
                    oldsig_direction_agreement_pct=float(100*np.mean((a[oldsig]>=0)==(b[oldsig]>=0))) if oldsig.any() else None,
                    bothsig_direction_agreement_pct=float(100*np.mean((a[oldsig&newsig]>=0)==(b[oldsig&newsig]>=0))) if (oldsig&newsig).any() else None)
                pixel_rows.append(row)
            if rule=='minus1_plus1':
                complete=np.isfinite(filtered['old']).all(axis=0)&np.isfinite(filtered['new']).all(axis=0)
                for scope,mask in scopes.items():
                    keep=complete&mask
                    for v in stacks:
                        means=np.average(filtered[v][:,keep],axis=1,weights=weights[keep])
                        x=YEARS[:nperiod]-YEARS[:nperiod].mean()
                        fit=sm.OLS(means,sm.add_constant(x)).fit()
                        for lag in [0,1,2,3,4]:
                            model=fit if lag==0 else fit.get_robustcov_results(cov_type='HAC',maxlags=lag,use_correction=True,use_t=True)
                            lo,hi=model.conf_int()[1]
                            regional.append(dict(period=period,scope=scope,version=v,pixels=int(keep.sum()),method='OLS' if lag==0 else f'HAC{lag}',
                                slope=float(model.params[1]),p=float(model.pvalues[1]),ci_low=float(lo),ci_high=float(hi)))
                        for yr,mean in zip(YEARS[:nperiod],means):
                            trajectories.append(dict(period=period,scope=scope,version=v,file_year=int(yr),reference_year=int(yr-1),mean_TAC=float(mean)))
                if nperiod==20:
                    for v in stacks:
                        write_map(f'{v}_TAC_OLS_slope_2004_2023.tif',ids,np.where(complete,slope[v],np.nan))
                    write_map('new_minus_old_TAC_slope_2004_2023.tif',ids,np.where(complete,slope['new']-slope['old'],np.nan))
            print(f'Computed {period}, {rule}',flush=True)
    pd.DataFrame(pixel_rows).to_csv(OUT/'pixel_trend_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(regional).to_csv(OUT/'regional_trends_OLS_HAC.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(trajectories).to_csv(OUT/'regional_trajectories.csv',index=False,encoding='utf-8-sig')
    summary=dict(new_files=21,file_endpoint_years=YEARS.tolist(),reference_years=(YEARS-1).tolist(),base_pixels=len(ids),
        original_RF_reproduction=reproduction,
        annual_bounded_summary=annual.query("rule=='both_in_minus1_plus1'").groupby('scope')[['pearson_r','MAE','bias_new_minus_old']].agg(['min','mean','max']).to_dict(),
        notes=['Original comparison is against the later same-filename TIFFs used by RF, not a claim of reproducing archived FIG1.',
            'Both pixelwise available-case fits and regional common-complete-support fits are reported separately.',
            'Regional means use cosine-latitude weights, constant support within period, retained TAC [-1,1]; they are paired diagnostics, not reconstructed manuscript FIG1 anomaly trajectories.',
            'RF trend check uses TAC [0,1], 19 endpoint years2005-2023, finite n>2, OLS with intercept, original P<0.05 labels.',
            '2025 endpoint adds reference year2024 and is an edge sensitivity, not automatically promoted into primary period.',
            'No original response, manuscript, RF model or lag-sensitivity result has been overwritten.'])
    # Tuple column names from pandas are represented explicitly for JSON.
    summary['annual_bounded_summary']={str(k):v for k,v in summary['annual_bounded_summary'].items()}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(pixel_rows).query("rule=='zero_plus1_RF_rule' and period=='file2005_2023_original_RF'").to_string(index=False),flush=True)
    print(pd.DataFrame(regional).query("period=='file2005_2024_lag_common' and method=='HAC4'").to_string(index=False),flush=True)


if __name__=='__main__':
    main()
