// Offline tests for the actual pure arithmetic/configuration functions.
// This is NOT an Earth Engine backend or export integration test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const filename = path.join(__dirname, 'nonERA5_covariates_public_005_GEE.js');
const source = fs.readFileSync(filename, 'utf8');
new vm.Script(source, {filename});
const ctx = vm.createContext({});
vm.runInContext(source.split('// EXECUTE:')[0], ctx);

class Scalar {
  constructor(value, valid = true) {this.value = value; this.valid = valid;}
  op(x, fn) {
    const y = x instanceof Scalar ? x : new Scalar(x);
    return new Scalar(fn(this.value, y.value), this.valid && y.valid);
  }
  add(x) {return this.op(x, (a,b) => a+b);}
  multiply(x) {return this.op(x, (a,b) => a*b);}
  divide(x) {return this.op(x, (a,b) => a/b);}
  pow(x) {return this.op(x, (a,b) => a**b);}
  tanh() {return new Scalar(Math.tanh(this.value), this.valid);}
  gte(x) {return this.op(x, (a,b) => a>=b);}
  lte(x) {return this.op(x, (a,b) => a<=b);}
  and(x) {return this.op(x, (a,b) => Boolean(a && b));}
  updateMask(x) {return new Scalar(this.value, this.valid && x.valid && Boolean(x.value));}
  rename() {return this;}
  set() {return this;}
}
let passed = 0;
function test(name, fn) {fn(); passed++; console.log('PASS: ' + name);}
function near(actual, expected) {assert.ok(Math.abs(actual-expected)<1e-12);}

test('default configuration and exactly four soil predictors', () => {
  ctx.validateConfig();
  assert.equal(ctx.SOILS.length,4);
  assert.deepEqual(Array.from(ctx.GRID),[0.05,0,-180,0,-0.05,80]);
});
test('thickness weights sum to 200 cm', () => near(Array.from(ctx.WEIGHTS).reduce((a,b)=>a+b),200));
test('depth-weighted mean differs from unweighted mean', () => {
  near(ctx.depthMean([1,2,3,4,5,6].map(x=>new Scalar(x))).value,4.95);
});
test('constant depth profile is preserved', () => {
  near(ctx.depthMean([8,8,8,8,8,8].map(x=>new Scalar(x))).value,8);
});
test('missing depth does not silently become zero or a shallower profile', () => {
  const layers = [1,2,3,4,5,6].map(x=>new Scalar(x));
  layers[5].valid = false;
  assert.equal(ctx.depthMean(layers).valid,false);
});
test('spatial aggregation uses the overlap-weighted mean, without extra pixelArea', () => {
  let reduction;
  const chain = new Proxy({}, {get(target,key) {
    if (key === 'reduceResolution') return options=>{reduction=options;return chain;};
    return ()=>chain;
  }});
  ctx.ee = {Reducer:{mean:()=> 'overlap-weighted-mean'}};
  const result = ctx.aggregateAreaMean(chain,'projection','test');
  assert.equal(reduction.reducer,'overlap-weighted-mean');
  assert.equal(reduction.bestEffort,false);
  assert.equal(result.value,chain);
  assert.equal(result.coverage,chain);
});
test('no vegetation configuration, dependencies or export paths remain', () => {
  assert.equal(/kndvi|modis|vegetation|summaryqa|dateaudit/i.test(source),false);
  assert.deepEqual(Object.keys(ctx.CFG),['driveFolder','soil','terrain','minSpatialCoverage',
    'exportShardSize','exportSmokeTest','formulaChecks']);
});
test('terrain source resolution remains explicit', () => {
  assert.ok(source.includes("source_resolution:'7.5 arc-sec'"));
});
test('invalid spatial coverage fails validation', () => {
  const original=ctx.CFG.minSpatialCoverage;
  [-1,1.1,NaN].forEach(v=>{
    ctx.CFG.minSpatialCoverage=v;
    assert.throws(()=>ctx.validateConfig(),/Invalid coverage/);
  });
  ctx.CFG.minSpatialCoverage=original;
});
test('cloud grid check reads exact export transform, without an unrelated scale', () => {
  const check=source.slice(source.indexOf('function numericalChecks('),source.indexOf('function main('));
  assert.ok(check.includes('crs:CRS,crsTransform:GRID,maxPixels:10'));
  assert.equal(/scale\s*:/.test(check),false);
  assert.ok(check.includes('overlap_coverage:0.75'));
  assert.ok(check.includes('CFG.minSpatialCoverage<=0.75 ? 3 : NODATA'));
});
test('failed checks expose plain-text values and still block task creation', () => {
  const check=source.slice(source.indexOf('function numericalChecks('),source.indexOf('function main('));
  assert.ok(check.includes("print('CHECK_DETAILS_JSON: '+JSON.stringify("));
  assert.ok(check.includes("print('GRID_DIAGNOSTICS_JSON: '+JSON.stringify("));
  const failure=check.slice(check.indexOf('if (!values || bad.length)'),check.indexOf("print('PASS: four numerical checks.'"));
  assert.ok(failure.includes('return;'));
  assert.equal(failure.includes('done();'),false);
});
test('synthetic source is geographically defined and checked before aggregation', () => {
  const check=source.slice(source.indexOf('function numericalChecks('),source.indexOf('function main('));
  assert.ok(check.includes('ee.Image.pixelLonLat()'));
  assert.equal(check.includes('ee.Image.pixelCoordinates('),false);
  assert.ok(check.includes("nativeSamples.aggregate_array('source_value').evaluate("));
  const gate=check.slice(check.indexOf('if (nativeError'),check.indexOf('function checkAggregated()'));
  assert.ok(gate.includes('return;'));
  assert.ok(gate.indexOf('return;')<gate.indexOf('checkAggregated();'));
  assert.ok(check.includes("print('NATIVE_CHECK_JSON: '+JSON.stringify("));
});
test('self-test uses aggregated target-grid band as first band', () => {
  const check=source.slice(source.indexOf('function numericalChecks('),source.indexOf('function main('));
  assert.ok(check.includes("var checks = aggregated.value.unmask(NODATA).rename('overlap_mean')"));
});
test('production aggregation avoids forced native-resolution reprojection', () => {
  const aggregate=source.slice(source.indexOf('function aggregateAreaMean('),source.indexOf('function rasterTask('));
  assert.ok(aggregate.includes("image.rename('value').setDefaultProjection(projection)"));
  assert.equal(aggregate.includes('reproject({crs:projection})'),false);
  assert.ok(aggregate.includes('.reproject({crs:CRS,crsTransform:GRID})'));
  const terrain=source.slice(source.indexOf('function prepareTerrain('),source.indexOf('function createTasks('));
  assert.ok(terrain.includes('ee.Terrain.slope(dem.setDefaultProjection(projection))'));
});
test('small shards are a compute setting, preserving the export grid', () => {
  assert.equal(ctx.CFG.exportShardSize,32);
  const task=source.slice(source.indexOf('function rasterTask('),source.indexOf('function prepareSoil('));
  assert.ok(task.includes('shardSize:CFG.exportShardSize'));
  assert.ok(task.includes('crs:CRS,crsTransform:GRID'));
  assert.equal(/\bscale\s*:/.test(task),false);
  assert.equal(task.includes('fileDimensions'),false);
  ctx.CFG.exportShardSize=33;
  assert.throws(()=>ctx.validateConfig(),/Invalid compute shard/);
  ctx.CFG.exportShardSize=32;
});
test('smoke region is small, grid-aligned, and does not clip production inputs', () => {
  assert.equal(ctx.CFG.exportSmokeTest,true);
  assert.ok(source.includes("'TEST_SOC_boreal_10E_15E_60N_65N'"));
  assert.ok(source.includes('[10+1e-6,60+1e-6,15-1e-6,65-1e-6]'));
  near((15-10)/ctx.GRID[0],100);
  near((65-60)/-ctx.GRID[4],100);
});
console.log(`${passed} offline tests passed. Cloud preflight, reducers and export execution remain untested.`);
