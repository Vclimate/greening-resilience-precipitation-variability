// Scalar-band mock for testing ACTUAL script arithmetic without GEE credentials.
// Does not emulate projection, raster resampling, backend types or exports.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Image {
  constructor(bands, props={}) {
    this.bands = Object.fromEntries(Object.entries(bands).map(([k,v])=>[k,
      typeof v === 'object' ? {...v} : {value:v,valid:true}]));
    this.props = props;
  }
  select(names) {
    names = typeof names === 'string' ? [names] : names;
    return new Image(Object.fromEntries(names.map(n=>[n,this.bands[n]])),this.props);
  }
  rename(names) {
    names = typeof names === 'string' ? [names] : names;
    return new Image(Object.fromEntries(Object.values(this.bands).map((v,i)=>[names[i],v])),this.props);
  }
  op(other, fn) {
    const right = other instanceof Image ? Object.values(other.bands) : [{value:other,valid:true}];
    return new Image(Object.fromEntries(Object.entries(this.bands).map(([n,b],i)=>{
      const y = right.length===1 ? right[0] : right[i];
      return [n,{value:fn(b.value,y.value),valid:b.valid && y.valid}];
    })),this.props);
  }
  unary(fn) {return this.op(0,a=>fn(a));}
  multiply(x) {return this.op(x,(a,b)=>a*b);}
  divide(x) {return this.op(x,(a,b)=>a/b);}
  add(x) {return this.op(x,(a,b)=>a+b);}
  subtract(x) {return this.op(x,(a,b)=>a-b);}
  eq(x) {return this.op(x,(a,b)=>Number(a===b));}
  gt(x) {return this.op(x,(a,b)=>Number(a>b));}
  gte(x) {return this.op(x,(a,b)=>Number(a>=b));}
  lt(x) {return this.op(x,(a,b)=>Number(a<b));}
  lte(x) {return this.op(x,(a,b)=>Number(a<=b));}
  and(x) {return this.op(x,(a,b)=>Number(Boolean(a&&b)));}
  not() {return this.unary(x=>Number(!x));}
  exp() {return this.unary(Math.exp);}
  abs() {return this.unary(Math.abs);}
  where(condition, other) {return Object.values(condition.bands)[0].value ? other : this;}
  updateMask(mask) {
    const masks = Object.values(mask.bands);
    return new Image(Object.fromEntries(Object.entries(this.bands).map(([name,b],i)=>{
      const m = masks.length===1 ? masks[0] : masks[i];
      return [name,{value:b.value,valid:b.valid && m.valid && Boolean(m.value)}];
    })),this.props);
  }
  mask() {return new Image(Object.fromEntries(Object.entries(this.bands).map(([n,b])=>[n,Number(b.valid)])));}
  addBands(other) {return new Image({...this.bands,...other.bands},this.props);}
  set(key,value) {return new Image(this.bands,{...this.props,[key]:value});}
  get(key) {return this.props[key];}
  date() {return new FakeDate(this.props.date);}
  toDouble() {return this;}
  setDefaultProjection() {return this;}
}
class FakeDate {
  constructor(date) {this.value=date;}
  advance(n,unit) {
    assert.equal(unit,'month');
    const d = this.value;
    return new FakeDate(new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+n,1)));
  }
  difference(other,unit) {assert.equal(unit,'day');return (this.value-other.value)/86400000;}
  millis() {return +this.value;}
}
class Collection {
  constructor(images) {this.images=images;}
  select(names) {return new Collection(this.images.map(i=>i.select(names)));}
  map(fn) {return new Collection(this.images.map(fn));}
  stats(fn) {
    const names=Object.keys(this.images[0].bands);
    return new Image(Object.fromEntries(names.map(name=>{
      const values=this.images.map(i=>i.bands[name]).filter(b=>b.valid).map(b=>b.value);
      return [name,{value:fn(values),valid:values.length>0}];
    })));
  }
  sum() {return this.stats(a=>a.reduce((s,v)=>s+v,0));}
  mean() {return this.stats(a=>a.reduce((s,v)=>s+v,0)/a.length);}
  count() {return this.stats(a=>a.length);}
  reduce(reducer) {
    if (reducer==='sampleStdDev') return this.stats(a=>{
      const mean=a.reduce((s,v)=>s+v,0)/a.length;
      return Math.sqrt(a.reduce((s,v)=>s+(v-mean)**2,0)/(a.length-1));
    });
    assert.equal(reducer,'linearFit');
    const rows=this.images.map(i=>Object.values(i.bands)).filter(row=>row.every(b=>b.valid));
    const mx=rows.reduce((s,r)=>s+r[0].value,0)/rows.length;
    const my=rows.reduce((s,r)=>s+r[1].value,0)/rows.length;
    const num=rows.reduce((s,r)=>s+(r[0].value-mx)*(r[1].value-my),0);
    const den=rows.reduce((s,r)=>s+(r[0].value-mx)**2,0);
    return new Image({scale:num/den});
  }
}
const ee={Image:{constant:v=>new Image({constant:v})},
  Number:v=>({subtract:other=>v-other}),
  Reducer:{sampleStdDev:()=> 'sampleStdDev',linearFit:()=> 'linearFit'}};
const ctx=vm.createContext({ee});
const script=fs.readFileSync(path.join(__dirname,'ERA5_all_covariates_bilinear_005_GEE.js'),'utf8');
new vm.Script(script);
vm.runInContext(script.split('// EXECUTE:')[0],ctx);
function raw(month=0,changes={}) {
  const date=new Date(Date.UTC(2020,month,1));
  const days=new Date(Date.UTC(2020,month+1,0)).getUTCDate();
  return new Image({total_precipitation_sum:0.1,total_evaporation_sum:-0.03,
    temperature_2m:293.15,dewpoint_temperature_2m:283.15,
    surface_solar_radiation_downwards_sum:15000000*days,
    volumetric_soil_water_layer_1:0.2,volumetric_soil_water_layer_2:0.4,...changes},{date});
}
function near(a,b) {assert.ok(Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(b)),`${a} != ${b}`);}
function value(image,band) {assert.equal(image.bands[band].valid,true);return image.bands[band].value;}
let passed=0;
function test(name,fn) {fn();passed++;console.log('PASS: '+name);}
test('monthly P and ET conversion, evaporation sign and P-ET',()=>{
  const x=ctx.monthlyFields(raw());
  near(value(x,'P'),100);near(value(x,'ET'),30);near(value(x,'P_ET'),70);
});
test('radiation monthly sum converted to daily amount, including leap February',()=>{
  [0,1,3].forEach(m=>near(value(ctx.monthlyFields(raw(m)),'Rs'),15000000));
});
test('temperature Celsius and thickness-weighted SM',()=>{
  const x=ctx.monthlyFields(raw());near(value(x,'Ta'),20);near(value(x,'SM'),0.35);
});
test('six VPD cases use the air-temperature phase',()=>{
  [[20,10],[-10,-15],[0,-5],[5,-5],[15,15],[-20,-25]].forEach(([air,dew])=>{
    const a=air>=0?17.67:22.46,b=air>=0?243.5:272.62;
    const expected=6.112*(Math.exp(a*air/(air+b))-Math.exp(a*dew/(dew+b)));
    near(value(ctx.vpdCelsius(new Image({t:air}),new Image({t:dew})),'VPD'),expected);
  });
});
test('condensation is retained with correct P-ET sign',()=>{
  const x=ctx.monthlyFields(raw(0,{total_evaporation_sum:0.002}));
  near(value(x,'ET'),-2);near(value(x,'P_ET'),102);near(value(x,'net_condensation'),1);
});
test('negative P is masked and counted without invalidating unrelated Ta',()=>{
  const x=ctx.monthlyFields(raw(0,{total_precipitation_sum:-0.1}));
  assert.equal(x.bands.P.valid,false);assert.equal(x.bands.P_ET.valid,false);
  near(value(x,'negative_P'),1);near(value(x,'Ta'),20);
});
test('invalid SM layer cannot be hidden by a plausible weighted mean',()=>{
  const x=ctx.monthlyFields(raw(0,{volumetric_soil_water_layer_1:-0.1}));
  assert.equal(x.bands.SM.valid,false);near(value(x,'invalid_SM'),1);
});
function year(amounts) {
  return ctx.annualFields(new Collection(amounts.map((p,m)=>ctx.monthlyFields(raw(m,{total_precipitation_sum:p/1000})))),'native');
}
test('annual totals and state means are not confused',()=>{
  const x=year(Array(12).fill(100));near(value(x,'P'),1200);near(value(x,'ET'),360);
  near(value(x,'P_ET'),840);near(value(x,'Ta'),20);near(value(x,'Rs'),15000000);
});
test('uniform monthly precipitation gives SI zero',()=>near(value(year(Array(12).fill(100)),'SI'),0));
test('single-month precipitation gives SI 11/6',()=>near(value(year([1200,...Array(11).fill(0)]),'SI'),11/6));
test('incomplete year is masked, not treated as a complete annual total',()=>{
  assert.equal(year(Array(11).fill(100)).bands.P.valid,false);
});
test('all-zero annual rainfall has valid P but undefined SI',()=>{
  const x=year(Array(12).fill(0));near(value(x,'P'),0);assert.equal(x.bands.SI.valid,false);
});
test('NIPV uses sample SD rather than population SD',()=>{
  near(value(ctx.sampleCV(new Collection([100,200,300].map(P=>new Image({P}))),'P',3),'P'),0.5);
});
test('zero-rainfall year retained in NIPV',()=>{
  near(value(ctx.sampleCV(new Collection([0,100,200].map(P=>new Image({P}))),'P',3),'P'),1);
});
test('zero period mean and incomplete baseline have undefined NIPV',()=>{
  assert.equal(ctx.sampleCV(new Collection([0,0,0].map(P=>new Image({P}))),'P',3).bands.P.valid,false);
  assert.equal(ctx.sampleCV(new Collection([100,200].map(P=>new Image({P}))),'P',3).bands.P.valid,false);
});
test('OLS uses actual years with centering, not array index gaps',()=>{
  const x=new Collection([2000,2001,2003].map(y=>new Image({P:100+2*(y-2000)},{year:y})));
  near(value(ctx.trendOf(x,'P',2000,'dP'),'dP'),2);
});
console.log(`${passed} ERA5 offline tests passed. No cloud export or spatial resampling test was performed.`);
