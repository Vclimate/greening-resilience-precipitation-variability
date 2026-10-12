// GEE Code Editor. Public datasets only; no private assets or uploads.
// Separate from ERA5_all_covariates_bilinear_005_GEE.js.
// Creates export tasks after preflight; start tasks manually in the Tasks tab.
// Terrain uses public GMTED2010 7.5 arc-sec, not a verified legacy 30 arc-sec DEM.

var CFG = {
  driveFolder: 'JRS_nonERA5_public_005_recomputed',
  soil: true,
  terrain: true,
  minSpatialCoverage: 0, // Mean over available source area; QA reports coverage.
  exportShardSize: 32, // Compute tile size, NOT output resolution or file size.
  exportSmokeTest: true, // Run the small SOC task before retrying global exports.
  formulaChecks: true
};
var CRS = 'EPSG:4326';
var GRID = [0.05, 0, -180, 0, -0.05, 80];
var BOUNDS = [-180+1e-6, -60+1e-6, 180-1e-6, 80-1e-6];
var NODATA = -9999;
var SCRIPT_VERSION = 'soil-terrain-small-shards-20260926-3';
var DEM_ID = 'USGS/GMTED2010_FULL';
var DEPTHS = ['0-5cm','5-15cm','15-30cm','30-60cm','60-100cm','100-200cm'];
var WEIGHTS = [5,10,15,30,40,100];
var SOILS = [
  {property:'clay',name:'Clay',unit:'g kg-1',max:1000},
  {property:'sand',name:'Sand',unit:'g kg-1',max:1000},
  {property:'soc',name:'SOC',unit:'dg kg-1',max:10000},
  {property:'nitrogen',name:'Nitrogen',unit:'cg kg-1',max:100000}
];

function validateConfig() {
  if (typeof CFG.minSpatialCoverage !== 'number' || !isFinite(CFG.minSpatialCoverage) ||
      CFG.minSpatialCoverage < 0 || CFG.minSpatialCoverage > 1) throw new Error('Invalid coverage.');
  if (!CFG.soil && !CFG.terrain) throw new Error('Enable at least one data group.');
  if ([16,32,64,128,256].indexOf(CFG.exportShardSize)<0) throw new Error('Invalid compute shard size.');
}

function depthMean(layers) {
  // Start with a real source band to preserve projection and intersect all masks.
  var total = layers[0].multiply(WEIGHTS[0]);
  for (var j=1;j<WEIGHTS.length;j++) total = total.add(layers[j].multiply(WEIGHTS[j]));
  return total.divide(WEIGHTS.reduce(function(a,b){return a+b;},0));
}

function aggregateAreaMean(image, projection, outputName) {
  // Preserve the native grid without forcing a large native-resolution tile.
  var source = image.rename('value').setDefaultProjection(projection);
  // reduceResolution already weights by mask * pixel-overlap fraction.
  // Multiplying by native pixelArea again would introduce an extra area weight.
  var valid = source.mask().gt(0).unmask({value:0,sameFootprint:false}).rename('coverage');
  // Let reduceResolution request native inputs for each small output shard.
  var reduced = source.addBands(valid)
    .reduceResolution({reducer:ee.Reducer.mean(),maxPixels:65536,bestEffort:false})
    .reproject({crs:CRS,crsTransform:GRID});
  var coverage = reduced.select('coverage').rename(outputName+'_coverage');
  var mean = reduced.select('value').updateMask(coverage.gt(0))
    .updateMask(coverage.gte(CFG.minSpatialCoverage)).rename(outputName);
  return {value:mean.toFloat(),coverage:coverage.toFloat()};
}

function rasterTask(image, name, metadata, bounds) {
  var exportName = name+'_v3';
  Export.image.toDrive({
    image:image.toFloat().set(metadata || {}).set({script_version:SCRIPT_VERSION,
      compute_shard_size:CFG.exportShardSize}).unmask({value:NODATA,sameFootprint:false}),
    description:exportName,folder:CFG.driveFolder,fileNamePrefix:exportName,
    region:ee.Geometry.Rectangle(bounds || BOUNDS,CRS,false),crs:CRS,crsTransform:GRID,
    maxPixels:1e9,shardSize:CFG.exportShardSize,fileFormat:'GeoTIFF',
    formatOptions:{cloudOptimized:true,noData:NODATA}
  });
}

function prepareSoil(definition) {
  var raw = ee.Image('projects/soilgrids-isric/'+definition.property+'_mean');
  var names = DEPTHS.map(function(d){return definition.property+'_'+d+'_mean';});
  var projection = raw.select(names[0]).projection();
  var layers = names.map(function(name) {
    var layer = raw.select(name).toDouble();
    return layer.updateMask(layer.gte(0).and(layer.lte(definition.max)));
  });
  // Retain stored physical units matching the current Table S1, no /10 here.
  var weighted = depthMean(layers).rename(definition.name);
  var result = aggregateAreaMean(weighted,projection,definition.name);
  return {result:result,metadata:{variable:definition.name,dataset:raw.id(),
    units:definition.unit,depth_cm:'0-200',depth_weights:JSON.stringify(WEIGHTS),
    spatial_processing:'area-weighted mean of valid native pixels',
    temporal_processing:'static',legacy_equivalence:'depth formula matches supplied GEE code; old MAT lineage not verified'}};
}

function prepareTerrain() {
  var dem = ee.Image(DEM_ID).select('mea').toFloat();
  // Broad physical range preserves real below-sea-level land while excluding fills.
  dem = dem.updateMask(dem.gte(-500).and(dem.lte(9000)));
  var projection = dem.projection();
  // Compute slope BEFORE any coarsening. Slope is scale-dependent.
  var slope = ee.Terrain.slope(dem.setDefaultProjection(projection));
  return {elevation:aggregateAreaMean(dem,projection,'Elevation'),
    slope:aggregateAreaMean(slope,projection,'Slope'),projection:projection};
}

function createTasks() {
  var rows = [];
  var spatialQA = [];
  if (CFG.soil) SOILS.forEach(function(def) {
    var prepared = prepareSoil(def);
    rasterTask(prepared.result.value,'SoilGrids_'+def.name+'_0_200cm_005deg_area_mean',prepared.metadata);
    if (CFG.exportSmokeTest && def.property==='soc') {
      // 100 x 100 output pixels in a boreal region: tests high-latitude reprojection.
      // Do not clip native inputs, so edge aggregation uses the full source support.
      rasterTask(prepared.result.value,'TEST_SOC_boreal_10E_15E_60N_65N',prepared.metadata,
        [10+1e-6,60+1e-6,15-1e-6,65-1e-6]);
    }
    rows.push(ee.Feature(null,prepared.metadata));
    spatialQA.push(prepared.result.coverage);
  });
  if (CFG.terrain) {
    var terrain = prepareTerrain();
    var meta = {dataset:DEM_ID,source_band:'mea',source_resolution:'7.5 arc-sec',
      spatial_processing:'area-weighted mean; slope calculated at native DEM resolution',
      temporal_processing:'static',legacy_equivalence:'same GMTED family; old 30 arc-sec processing NOT reproduced'};
    rasterTask(terrain.elevation.value,'GMTED2010_mea7p5as_Elevation_005deg_area_mean',meta);
    rasterTask(terrain.slope.value,'GMTED2010_mea7p5as_Slope_005deg_area_mean',meta);
    rows.push(ee.Feature(null,meta).set({variable:'Elevation',units:'m'}));
    rows.push(ee.Feature(null,meta).set({variable:'Slope',units:'degree'}));
    spatialQA.push(terrain.elevation.coverage);
    spatialQA.push(terrain.slope.coverage);
  }
  if (spatialQA.length) rasterTask(ee.Image.cat(spatialQA),'QA_nonERA5_native_area_coverage_005deg',
    {units:'fraction',meaning:'valid native area / total target area; zero coverage is not a valid predictor'});
  var globalMeta = {target_crs:CRS,target_transform:JSON.stringify(GRID),nodata:NODATA,
    expected_rows:2800,expected_columns:7200,min_spatial_coverage:CFG.minSpatialCoverage,
    script_version:SCRIPT_VERSION,compute_shard_size:CFG.exportShardSize};
  var manifest = ee.FeatureCollection(rows).map(function(feature){return feature.set(globalMeta);});
  Export.table.toDrive({collection:manifest,description:'nonERA5_public_covariates_metadata_v3',
    folder:CFG.driveFolder,fileNamePrefix:'nonERA5_public_covariates_metadata_v3',fileFormat:'CSV'});
  print('Tasks created, NOT started. Default: 6 predictor TIFs + coverage QA + metadata CSV.');
  if (CFG.soil && CFG.exportSmokeTest) print('Additional small TEST_SOC task: run this FIRST; retry global SOC only after it succeeds.');
  print('Compute shard size:',CFG.exportShardSize,'Output grid unchanged:',GRID);
  print('Enabled groups:',{soil:CFG.soil,terrain:CFG.terrain});
  print('Excluded: original ESA-CCI Forest_fraction, Mu HFP, WTD and Species; no verified matching public assets.');
  print('WARNING: terrain slope uses 7.5 arc-sec DEM, not the unverified legacy 30 arc-sec processing.');
  print('Do not replace RF inputs automatically. Verify downloaded data and metadata first.');
}

function numericalChecks(done) {
  var depth = depthMean([1,2,3,4,5,6].map(function(v){return ee.Image.constant(v).toDouble();}));
  // Four subpixels: mask value 1, leaving mean 3 and coverage 0.75.
  // Define values geographically rather than assuming pixelCoordinates indices.
  var fineGrid = [0.025,0,10,0,-0.025,0.05];
  var fine = ee.Projection(CRS,fineGrid);
  var lonlat = ee.Image.pixelLonLat().reproject({crs:CRS,crsTransform:fineGrid});
  var east = lonlat.select('longitude').gte(10.025);
  var south = lonlat.select('latitude').lt(0.025);
  var four = east.add(south.multiply(2)).add(1).toDouble()
    .updateMask(east.or(south)).reproject({crs:CRS,crsTransform:fineGrid});
  var nativeSamples = four.rename('source_value').unmask(NODATA).sample({
    region:ee.Geometry.Rectangle([10,0,10.05,0.05],CRS,false),
    projection:fine,geometries:true,dropNulls:false
  });
  // A fixture error must not be confused with a production aggregation error.
  nativeSamples.aggregate_array('source_value').evaluate(function(nativeValues,nativeError) {
    var expectedNative = [NODATA,2,3,4];
    var actualNative = nativeValues ? nativeValues.slice().sort(function(a,b){return a-b;}) : null;
    print('NATIVE_CHECK_JSON: '+JSON.stringify({version:SCRIPT_VERSION,
      expected:expectedNative,actual:actualNative,error:nativeError || null}));
    if (nativeError || JSON.stringify(actualNative)!==JSON.stringify(expectedNative)) {
      print('STOP: synthetic source pixels are incorrect. No tasks created.');
      return;
    }
    print('PASS: native four-pixel fixture.',actualNative);
    checkAggregated();
  });
  function checkAggregated() {
  var aggregated = aggregateAreaMean(four,fine,'test_mean');
  // Keep the export-grid image as the first band, not a global 1-degree constant.
  var checks = aggregated.value.unmask(NODATA).rename('overlap_mean')
    .addBands(aggregated.coverage.rename('overlap_coverage'))
    .addBands(depth.rename('weighted_depth'))
    .addBands(ee.Image.constant(5).updateMask(ee.Image.constant(0)).unmask(NODATA).rename('missing_not_zero'));
  checks.reduceRegion({reducer:ee.Reducer.first(),geometry:ee.Geometry.Point([10.025,0.025]),
    // Read the exact export grid, not an unrelated 1000-m sampling grid.
    crs:CRS,crsTransform:GRID,maxPixels:10}).evaluate(function(values,error) {
    var expected = {weighted_depth:4.95,missing_not_zero:NODATA,
      overlap_mean:CFG.minSpatialCoverage<=0.75 ? 3 : NODATA,overlap_coverage:0.75};
    var bad = [];
    if (values) Object.keys(expected).forEach(function(k) {
      if (typeof values[k] !== 'number' || !isFinite(values[k]) || Math.abs(values[k]-expected[k])>1e-5) bad.push(k);
    });
    if (error) {
      print('STOP: formula checks could not be evaluated. No tasks created.');
      print('CHECK_ERROR: '+String(error));
      return;
    }
    if (!values || bad.length) {
      print('STOP: formula checks failed. No tasks created.',{failed:bad,expected:expected,actual:values});
      // Plain strings remain visible without expanding Code Editor objects.
      print('CHECK_DETAILS_JSON: '+JSON.stringify({version:SCRIPT_VERSION,failed:bad,expected:expected,actual:values}));
      Object.keys(expected).forEach(function(k) {
        var actual = values ? values[k] : undefined;
        print('CHECK '+k+': expected='+expected[k]+', actual='+String(actual));
      });
      print('Grid diagnostics',{source:four.projection(),target:aggregated.value.projection(),
        export_crs:CRS,export_transform:GRID});
      print('Native four-pixel diagnostic (expected values -9999, 2, 3, 4)',nativeSamples);
      ee.Dictionary({source_projection:four.projection(),target_projection:aggregated.value.projection(),
        native_samples:nativeSamples}).evaluate(function(diagnostics,diagnosticError) {
        if (diagnosticError) print('GRID_DIAGNOSTIC_ERROR: '+String(diagnosticError));
        else print('GRID_DIAGNOSTICS_JSON: '+JSON.stringify(diagnostics));
      });
      return;
    }
    print('PASS: four numerical checks.',values);
    done();
  });
  }
}

function main() {
  validateConfig();
  print('SCRIPT_VERSION: '+SCRIPT_VERSION);
  var checks = {};
  if (CFG.soil) SOILS.forEach(function(def) {
    checks['soil_'+def.property] = ee.Image('projects/soilgrids-isric/'+def.property+'_mean').bandNames();
  });
  if (CFG.terrain) checks.dem_bands = ee.Image(DEM_ID).bandNames();
  ee.Dictionary(checks).evaluate(function(info,error) {
    if (error || !info) {print('STOP: public dataset preflight failed. No tasks created.',error);return;}
    var problems = [];
    if (CFG.soil) SOILS.forEach(function(def) {
      DEPTHS.forEach(function(d) {
        if (info['soil_'+def.property].indexOf(def.property+'_'+d+'_mean')<0) problems.push('Missing soil band '+def.property+' '+d);
      });
    });
    if (CFG.terrain && info.dem_bands.indexOf('mea')<0) problems.push('Missing DEM mea band');
    if (problems.length) {print('STOP: preflight issues.',problems,info);return;}
    print('PASS: soil/terrain public assets and expected bands.',info);
    if (CFG.formulaChecks) numericalChecks(createTasks); else createTasks();
  });
}

// EXECUTE: source preflight and numerical checks precede all export creation.
main();
