"""Small reducer conformance test; no exports or global GEE TAC jobs."""
import json
from pathlib import Path
import ee
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output/kNDVI_TAC_local_QA0_20261003"


def main():
    ee.Initialize(project="resilience20260619")
    ee.data.setDeadline(120000)
    cases, definitions = {}, {}
    for p in range(5):
        data = np.load(OUT/f"point_{p}_stages.npz")
        dates,res=data["dates"],data["residual"]
        for endpoint in [2006,2008]:
            label=f"point{p}_{endpoint}"
            idx=np.flatnonzero((dates>=np.datetime64(f"{endpoint-5}-01-01"))&(dates<np.datetime64(f"{endpoint}-01-01")))
            x,y=res[idx[:-1]].astype(float),res[idx[1:]].astype(float)
            valid=np.isfinite(x)&np.isfinite(y)
            x,y=x[valid],y[valid]
            design=np.column_stack([np.ones(len(x)),x])
            ols=np.linalg.lstsq(design,y,rcond=None)[0]
            err=y-design@ols
            sigma=float(np.sqrt(np.mean(err*err)))
            mad=float(np.median(np.abs(err-np.median(err)))/.6744897501960817)
            med=float(np.median(np.abs(err)))
            fc=ee.FeatureCollection([ee.Feature(None,{"one":1,"x":float(a),"y":float(b)}) for a,b in zip(x,y)])
            result={"default":fc.reduceColumns(ee.Reducer.robustLinearRegression(2,1),["one","x","y"])}
            margins={"RMSE_1.345":1.345*sigma,"RMSE_2":2*sigma,"RMSE_2.795":2.795*sigma,
                     "MAD_2.795":2.795*mad,"MED_2.795":2.795*med}
            for factor in [1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
                margins[f"MED_{factor}"]=factor*med
            for name,margin in margins.items():
                result[name]=fc.reduceColumns(ee.Reducer.robustLinearRegression(2,1,float(margin)),["one","x","y"])
            cases[label]=ee.Dictionary(result)
            definitions[label]={"x":x.tolist(),"y":y.tolist(),"OLS":ols.tolist(),"margins":margins}
    result=ee.Dictionary(cases).getInfo()
    report={"cases":result,"definitions":definitions}
    (OUT/"robust_oracle_probe.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    for name,result in result.items():
        expected=np.array(result["default"]["coefficients"]).ravel()
        matches=[k for k,v in result.items() if k!="default" and np.max(np.abs(np.array(v["coefficients"]).ravel()-expected))<1e-12]
        print(name,"default",expected,"exact_margin_matches",matches,flush=True)


if __name__=="__main__":
    main()
