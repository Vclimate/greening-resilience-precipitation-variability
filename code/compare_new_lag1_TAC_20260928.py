"""Compare two new lag-1 exports to the original same-window TAC, without resampling."""
import json
import os
from pathlib import Path
import sys

DLL=os.add_dll_directory(str(Path(sys.base_prefix)/'Library/bin'))
import pyarrow.parquet
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from audit_GEE_exports_pre_RF import ROOT,DEST,digest

OUT=ROOT/'output/TAC_lag1_new_vs_original_20260928'
OLD=ROOT/'input/KNDVI_AC1_5year_window-01-01_005'
NEW=Path('C:/Users/user/Downloads')


def summarize(a,b):
    delta=b-a
    return dict(paired=int(len(a)),old_mean=float(a.mean()),new_mean=float(b.mean()),
        old_median=float(np.median(a)),new_median=float(np.median(b)),
        pearson_r=float(np.corrcoef(a,b)[0,1]),bias_new_minus_old=float(delta.mean()),
        MAE=float(np.mean(np.abs(delta))),RMSE=float(np.sqrt(np.mean(delta**2))),
        median_absolute_difference=float(np.median(np.abs(delta))),
        p95_absolute_difference=float(np.quantile(np.abs(delta),.95)),
        p99_absolute_difference=float(np.quantile(np.abs(delta),.99)),
        max_absolute_difference=float(np.max(np.abs(delta))),
        percent_absdiff_lt_0_01=float(100*np.mean(np.abs(delta)<.01)),
        percent_absdiff_gt_0_05=float(100*np.mean(np.abs(delta)>.05)),
        percent_absdiff_gt_0_1=float(100*np.mean(np.abs(delta)>.1)),
        percent_equal_at_1e_6=float(100*np.mean(np.abs(delta)<=1e-6)),
        sign_disagreement_percent=float(100*np.mean((a>=0)!=(b>=0))))


def readmat(s):
    with h5py.File(s['path'],'r') as f:
        assert f[s['dataset']].shape==(7200,3600)
        return f[s['dataset']][:,200:3000].T


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DEST/'RF_input_manifest_22.json').read_text(encoding='utf-8'))
    support={s['variable']:s for s in manifest['response_and_masks']}
    lc=readmat(support['land_cover'])
    stable=readmat(support['stable_vegetation_mask'])
    mdn=readmat(support['MDN'])
    y=readmat(support['delta_TAC'])
    base=(stable==1)&np.isfinite(lc)&(lc!=0)&np.isfinite(mdn)&(mdn<.5)&np.isfinite(y)
    masks={'All_export_pixels':np.ones((2800,7200),bool),
           'Original_stable_vegetation':base,
           'Forest':base&(lc<=5),'Non-forest':base&(lc>5)}
    del stable,mdn,y,lc
    summaries=[];coverage=[];metadata=[];series={};sensitivity=[];extremes=[]
    for year in [2006,2008]:
        paths=[OLD/f'KNDVI_AC1_5year_window_{year}_01_01.tif',NEW/f'KNDVI_TAC_lag1_5year_window_{year}_01_01.tif']
        arrays=[]
        profiles=[]
        for label,path in zip(['old','new'],paths):
            with rasterio.open(path) as ds:
                profiles.append((ds.crs,ds.transform))
                a=ds.read(1,masked=True,window=Window(0,0,7200,2800)).astype('float64').filled(np.nan)
                fill=(np.abs(a)>1e30)|np.isin(a,[-9999,-32768])
                a[fill|~np.isfinite(a)]=np.nan
                item=dict(year=year,version=label,path=str(path),sha256=digest(path),
                    shape=list(ds.shape),crs=str(ds.crs),transform=list(ds.transform),
                    nodata=ds.nodata,band=ds.descriptions[0],
                    finite_count=int(np.isfinite(a).sum()),finite_zero=int((a==0).sum()),
                    finite_min=float(np.nanmin(a)),finite_max=float(np.nanmax(a)),
                    outside_minus1_plus1=int((np.abs(a)>1).sum()))
                if label=='new':
                    assert ds.shape==(2801,7201)
                    edge_row=ds.read(1,masked=True,window=Window(0,2800,7201,1)).astype(float).filled(np.nan)
                    edge_col=ds.read(1,masked=True,window=Window(7200,0,1,2800)).astype(float).filled(np.nan)
                    edge=np.concatenate([edge_row.ravel(),edge_col.ravel()])
                    item['extra_south_east_finite_count']=int(np.isfinite(edge).sum())
                else:
                    assert ds.shape==(2800,7200)
                arrays.append(a);metadata.append(item)
        assert profiles[0][0]==profiles[1][0]
        np.testing.assert_allclose(list(profiles[0][1]),list(profiles[1][1]),rtol=0,atol=1e-10)
        a,b=arrays
        for scope,mask in masks.items():
            va=mask&np.isfinite(a);vb=mask&np.isfinite(b);both=va&vb
            row=dict(year=year,scope=scope,scope_pixels=int(mask.sum()),old_valid=int(va.sum()),
                     new_valid=int(vb.sum()),lost=int((va&~vb).sum()),gained=int((vb&~va).sum()))
            coverage.append(row)
            result=dict(year=year,scope=scope,**summarize(a[both],b[both]))
            summaries.append(result)
            print(json.dumps(result),flush=True)
            conventional=both&(np.abs(a)<=1)&(np.abs(b)<=1)
            sensitivity.append(dict(year=year,scope=scope,
                old_abs_above_1=int((va&(np.abs(a)>1)).sum()),
                new_abs_above_1=int((vb&(np.abs(b)>1)).sum()),
                pair_excluded=int((both&~conventional).sum()),
                **summarize(a[conventional],b[conventional])))
        series[year]=(a[base],b[base])
        diff=b-a
        indices=np.flatnonzero(base&np.isfinite(diff))
        worst=indices[np.argsort(np.abs(diff.ravel()[indices]))[-20:][::-1]]
        for index in worst:
            rr,cc=divmod(int(index),7200)
            extremes.append(dict(year=year,longitude=-180+(cc+.5)*.05,latitude=80-(rr+.5)*.05,
                old_TAC=float(a[rr,cc]),new_TAC=float(b[rr,cc]),difference=float(diff[rr,cc])))
        with rasterio.open(OUT/f'TAC_{year}_new_minus_original.tif','w',driver='GTiff',
            height=2800,width=7200,count=1,dtype='float32',crs=profiles[0][0],
            transform=profiles[1][1],nodata=-9999,compress='deflate',tiled=True,predictor=3) as ds:
            ds.write(np.where(np.isfinite(diff),diff,-9999).astype('float32'),1)
            ds.set_band_description(1,'new_minus_original_TAC')
        del a,b,arrays,diff
    a1,b1=series[2006];a2,b2=series[2008]
    valid=np.isfinite(a1)&np.isfinite(a2)&np.isfinite(b1)&np.isfinite(b2)
    temporal=summarize((a2-a1)[valid],(b2-b1)[valid])
    temporal['meaning']='2008-endpoint TAC minus 2006-endpoint TAC, not full-period delta_TAC trend or significance.'
    bounded=valid&(np.abs(a1)<=1)&(np.abs(a2)<=1)&(np.abs(b1)<=1)&(np.abs(b2)<=1)
    temporal['sensitivity_all_four_TAC_within_minus1_plus1']=summarize((a2-a1)[bounded],(b2-b1)[bounded])
    pd.DataFrame(summaries).to_csv(OUT/'comparison_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(OUT/'mask_coverage.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity_both_TAC_within_minus1_plus1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(extremes).to_csv(OUT/'largest_differences_on_original_vegetation.csv',index=False,encoding='utf-8-sig')
    (OUT/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'two_window_change_check.json').write_text(json.dumps(temporal,indent=2),encoding='utf-8')
    print('COVERAGE',json.dumps(coverage),flush=True)
    print('TWO_WINDOW_CHANGE',json.dumps(temporal),flush=True)


if __name__=='__main__':
    main()
