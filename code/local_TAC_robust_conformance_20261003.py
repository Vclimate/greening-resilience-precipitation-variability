"""Independent synthetic conformance cases for the offline Talwar implementation."""
import json
from pathlib import Path
import ee
import numpy as np

OUT=Path(__file__).resolve().parents[1]/"output/kNDVI_TAC_local_QA0_20261003"


def talwar(x,y):
    design=np.column_stack([np.ones(len(x)),x])
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    beta=4*np.median(np.abs(y-design@coef))
    keep=np.ones(len(x),bool)
    iterations=0
    while True:
        next_keep=keep & (np.abs(y-design@coef)<=beta)
        if next_keep.sum()<3:
            return np.full(2,np.nan),beta,iterations
        if np.array_equal(next_keep,keep):
            break
        keep=next_keep
        coef=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
        iterations+=1
    return coef,beta,iterations


def main():
    ee.Initialize(project="resilience20260619")
    ee.data.setDeadline(120000)
    defs, cases={},{}
    for i in range(12):
        rng=np.random.default_rng(3800+i)
        n=31+i*7
        x=rng.normal(0,.04,n)
        y=.003+.6*x+rng.normal(0,.02,n)
        y[::13]+=rng.normal(0,.12,len(y[::13]))
        coef,beta,it=talwar(x,y)
        fc=ee.FeatureCollection([ee.Feature(None,{"one":1,"x":float(a),"y":float(b)}) for a,b in zip(x,y)])
        cases[str(i)]=fc.reduceColumns(ee.Reducer.robustLinearRegression(2,1),["one","x","y"])
        defs[str(i)]={"x":x.tolist(),"y":y.tolist(),"local":coef.tolist(),"beta":beta,"iterations":it}
    expected=ee.Dictionary(cases).getInfo()
    checks=[]
    for key,definition in defs.items():
        actual=np.array(definition["local"])
        target=np.array(expected[key]["coefficients"]).ravel()
        error=float(np.max(np.abs(actual-target)))
        checks.append(dict(case=key,error=error,expected=target.tolist(),**definition))
        print(key,"max_coef_error",error,flush=True)
    report={"rule":"fixed beta=4*median(abs(initial OLS residual)); permanently zero-weight excluded observations until unchanged",
            "note":"Empirical conformance evidence, not a published specification of the private GEE server implementation.",
            "checks":checks,"all_passed":all(c["error"]<1e-10 for c in checks)}
    (OUT/"robust_independent_conformance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    assert report["all_passed"], "Do not run full local reproduction until the implementation matches"


if __name__=="__main__":
    main()
