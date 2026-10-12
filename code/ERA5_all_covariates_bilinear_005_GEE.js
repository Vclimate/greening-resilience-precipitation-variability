// Standalone Google Earth Engine Code Editor script.
// Continuous final covariates: bilinear interpolation AFTER native-grid calculation.
// Discrete QA/count bands: nearest neighbor. Do not interpolate counts.
// Creates tasks only. Start them manually in the Tasks tab.

var CFG = {
  studyStart: 2000,
  studyEnd: 2023,
  baselineStart: 1990,
  baselineEnd: 2023,
  driveFolder: 'JRS_ERA5_all_bilinear_005_recomputed',
  formulaChecks: true,
  exportPointAudit: true,
  showMaps: false
};
var DATASET = 'ECMWF/ERA5_LAND/MONTHLY_AGGR';
var CRS = 'EPSG:4326';
var GRID = [0.05, 0, -180, 0, -0.05, 80];
// A tiny inset prevents floating-point region rounding from adding a row/column.
var REGION_BOUNDS = [-180 + 1e-6, -60 + 1e-6, 180 - 1e-6, 80 - 1e-6];
var NODATA = -9999;
var SOURCE_BANDS = [
  'total_precipitation_sum', 'total_evaporation_sum',
  'temperature_2m', 'dewpoint_temperature_2m',
  'surface_solar_radiation_downwards_sum',
  'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2'
];
var TOTAL_BANDS = ['P', 'ET', 'P_ET'];
var MEAN_BANDS = ['Ta', 'Rs', 'SM', 'VPD'];
var ANNUAL_BANDS = TOTAL_BANDS.concat(MEAN_BANDS);
var PREDICTORS = ['P', 'dP', 'P_SI', 'P_NIPV', 'Ta', 'dTa', 'VPD', 'SM', 'Rs', 'P_ET'];
var UNITS = {
  P: 'mm yr-1', dP: 'mm yr-2', P_SI: '1', P_NIPV: '1',
  Ta: 'degC', dTa: 'degC yr-1', VPD: 'hPa', SM: 'm3 m-3',
  Rs: 'J m-2 day-1', P_ET: 'mm yr-1', ET: 'mm yr-1'
};

function validateConfig() {
  [CFG.studyStart, CFG.studyEnd, CFG.baselineStart, CFG.baselineEnd].forEach(function(y) {
    if (y !== Math.floor(y) || y < 1950) throw new Error('Invalid integer year: ' + y);
  });
  if (CFG.studyEnd <= CFG.studyStart || CFG.baselineEnd <= CFG.baselineStart) {
    throw new Error('Each period must contain at least two years.');
  }
}

// Matches D:/Code_Model/19636059/ForestResilience/VPD/cal_VPD/cal_VPD.m.
// Both es and e use the phase selected by AIR temperature, not dewpoint.
function vpdCelsius(air, dew) {
  function pressure(temp, a, b) {
    return temp.multiply(a).divide(temp.add(b)).exp().multiply(6.112);
  }
  var water = pressure(air, 17.67, 243.5).subtract(pressure(dew, 17.67, 243.5));
  var ice = pressure(air, 22.46, 272.62).subtract(pressure(dew, 22.46, 272.62));
  return water.where(air.lt(0), ice).rename('VPD');
}

function monthlyFields(image) {
  var date = image.date();
  var days = date.advance(1, 'month').difference(date, 'day');
  var pRaw = image.select('total_precipitation_sum').multiply(1000).rename('P');
  var p = pRaw.updateMask(pRaw.gte(0));
  // ECMWF evaporation is normally negative. Keep signed condensation values.
  var et = image.select('total_evaporation_sum').multiply(-1000).rename('ET');
  var balance = p.subtract(et).rename('P_ET');
  var air = image.select('temperature_2m').subtract(273.15).rename('Ta');
  var dew = image.select('dewpoint_temperature_2m').subtract(273.15).rename('Td');
  var vpdRaw = vpdCelsius(air, dew);
  // Record unphysical negatives, then mask them rather than silently clipping to 0.
  var vpd = vpdRaw.updateMask(vpdRaw.gte(0));
  var rsRaw = image.select('surface_solar_radiation_downwards_sum').divide(days).rename('Rs');
  var rs = rsRaw.updateMask(rsRaw.gte(0));
  var sm1 = image.select('volumetric_soil_water_layer_1');
  var sm2 = image.select('volumetric_soil_water_layer_2');
  var smValid = sm1.gte(0).and(sm1.lte(1)).and(sm2.gte(0)).and(sm2.lte(1));
  var sm = sm1.multiply(7).add(sm2.multiply(21)).divide(28).rename('SM').updateMask(smValid);
  return p.addBands(et).addBands(balance).addBands(air).addBands(rs)
    .addBands(sm).addBands(vpd).addBands(dew)
    .addBands(pRaw.lt(0).rename('negative_P'))
    .addBands(vpdRaw.lt(0).rename('negative_VPD'))
    .addBands(rsRaw.lt(0).rename('negative_Rs'))
    .addBands(smValid.not().rename('invalid_SM'))
    .addBands(et.lt(0).rename('net_condensation'))
    .set('system:time_start', date.millis());
}

function annualFields(months, projection) {
  var counts = months.select(ANNUAL_BANDS).count();
  var sums = months.select(TOTAL_BANDS).sum().updateMask(counts.select(TOTAL_BANDS).eq(12));
  // Match legacy nanmean(monthly,3): equal monthly weights, not day-weighted.
  var means = months.select(MEAN_BANDS).mean().updateMask(counts.select(MEAN_BANDS).eq(12));
  var p = sums.select('P');
  var deviations = months.select('P').map(function(image) {
    return image.subtract(p.divide(12)).abs();
  }).sum();
  var si = deviations.divide(p.updateMask(p.gt(0))).rename('SI')
    .updateMask(counts.select('P').eq(12));
  return sums.addBands(means).addBands(si).setDefaultProjection(projection);
}

function trendOf(collection, band, firstYear, outputBand) {
  var xy = collection.map(function(image) {
    var y = image.select(band);
    var t = ee.Image.constant(ee.Number(image.get('year')).subtract(firstYear))
      .toDouble().rename('time').updateMask(y.mask());
    return t.addBands(y);
  });
  return xy.select(['time', band]).reduce(ee.Reducer.linearFit()).select('scale').rename(outputBand);
}

function sampleCV(collection, band, expectedCount) {
  var values = collection.select(band);
  var mean = values.mean();
  // Zero-rainfall years are valid CV observations; only a zero period mean is undefined.
  return values.reduce(ee.Reducer.sampleStdDev()).divide(mean.updateMask(mean.gt(0)))
    .updateMask(values.count().eq(expectedCount));
}

function nativeImage(image, projection) {
  return image.setDefaultProjection(projection).reproject({crs: projection});
}

function continuousToTarget(image, projection) {
  return nativeImage(image, projection).resample('bilinear')
    .reproject({crs: CRS, crsTransform: GRID}).toFloat();
}

function categoricalToTarget(image, projection) {
  return nativeImage(image, projection).reproject({crs: CRS, crsTransform: GRID}).toFloat();
}

function createRasterTask(image, label, first, last, method, region) {
  var name = 'ERA5Land_' + label + '_' + first + '_' + last + '_005deg_' + method;
  Export.image.toDrive({
    // Fill AFTER interpolation, otherwise -9999 contaminates adjacent pixels.
    image: image.unmask({value: NODATA, sameFootprint: false}),
    description: name, folder: CFG.driveFolder, fileNamePrefix: name,
    region: region, crs: CRS, crsTransform: GRID, maxPixels: 1e9,
    shardSize: 256, fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA}
  });
}

function numericalChecks(onSuccess) {
  var projection = ee.Projection('EPSG:4326').atScale(10000);
  function toyYear(amounts, year) {
    var images = amounts.map(function(value) {
      var p = ee.Image.constant(value).toDouble();
      return p.rename('P').addBands(p.rename('ET')).addBands(p.multiply(0).rename('P_ET'))
        .addBands(p.rename('Ta')).addBands(p.rename('Rs'))
        .addBands(p.rename('SM')).addBands(p.rename('VPD'));
    });
    return annualFields(ee.ImageCollection.fromImages(images), projection).set('year', year);
  }
  var ones = [1,1,1,1,1,1,1,1,1,1,1,1];
  var series = ee.ImageCollection.fromImages([1,2,3].map(function(v) {
    return toyYear(ones.map(function() {return v;}), v-1);
  }));
  var checks = toyYear(ones,0).select('SI').rename('uniform_SI')
    .addBands(toyYear([12,0,0,0,0,0,0,0,0,0,0,0],0).select('SI').rename('pulse_SI'))
    .addBands(series.select('P').reduce(ee.Reducer.sampleStdDev())
      .divide(series.select('P').mean()).rename('sample_CV'))
    .addBands(sampleCV(ee.ImageCollection.fromImages([0,1,2].map(function(v) {
      return toyYear(ones.map(function(){return v;}),v);
    })),'P',3).rename('CV_with_zero_year'))
    .addBands(trendOf(series,'P',0,'known_trend'))
    .addBands(toyYear(ones.slice(1),0).select('P').unmask(-9999).rename('missing_month'))
    .addBands(toyYear(ones.map(function(){return 0;}),0).select('SI').unmask(-9999).rename('zero_SI'))
    .addBands(vpdCelsius(ee.Image.constant(20),ee.Image.constant(10)).rename('warm_VPD'))
    .addBands(vpdCelsius(ee.Image.constant(-10),ee.Image.constant(-15)).rename('cold_VPD'))
    .addBands(vpdCelsius(ee.Image.constant(0),ee.Image.constant(-5)).rename('zeroC_VPD'));
  function pressure(t,a,b) {return 6.112*Math.exp(a*t/(t+b));}
  var expected = {
    uniform_SI:0, pulse_SI:11/6, sample_CV:0.5, CV_with_zero_year:1, known_trend:12,
    missing_month:-9999, zero_SI:-9999,
    warm_VPD:pressure(20,17.67,243.5)-pressure(10,17.67,243.5),
    cold_VPD:pressure(-10,22.46,272.62)-pressure(-15,22.46,272.62),
    zeroC_VPD:pressure(0,17.67,243.5)-pressure(-5,17.67,243.5)
  };
  checks.reduceRegion({reducer:ee.Reducer.first(),geometry:ee.Geometry.Point([0,0]),
    crs:projection,scale:10000,maxPixels:10}).evaluate(function(values,error) {
    var failed = [];
    if (!error && values) Object.keys(expected).forEach(function(k) {
      if (typeof values[k] !== 'number' || !isFinite(values[k]) ||
          Math.abs(values[k]-expected[k])>1e-5) failed.push(k);
    });
    if (error || !values || failed.length) {
      print('STOP: formula checks failed. No tasks created.',error,failed,values); return;
    }
    print('PASS: 10 numerical checks',values);
    onSuccess();
  });
}

function buildExports(raw, first, last) {
  var projection = ee.Image(raw.first()).select('total_precipitation_sum').projection();
  var region = ee.Geometry.Rectangle(REGION_BOUNDS, CRS, false);
  var monthly = raw.map(monthlyFields);
  var annual = ee.ImageCollection.fromImages(ee.List.sequence(first,last).map(function(year) {
    year = ee.Number(year);
    var date = ee.Date.fromYMD(year,1,1);
    return annualFields(monthly.filterDate(date,date.advance(1,'year')),projection)
      .set('year',year).set('system:time_start',date.millis());
  }));
  function period(collection, a, b) {
    return collection.filterDate(ee.Date.fromYMD(a,1,1),ee.Date.fromYMD(b+1,1,1));
  }
  var study = period(annual,CFG.studyStart,CFG.studyEnd);
  var baseline = period(annual,CFG.baselineStart,CFG.baselineEnd);
  var studyMonths = period(monthly,CFG.studyStart,CFG.studyEnd);
  var baselineMonths = period(monthly,CFG.baselineStart,CFG.baselineEnd);
  var ns = CFG.studyEnd-CFG.studyStart+1;
  var nb = CFG.baselineEnd-CFG.baselineStart+1;
  var studyCounts = study.select(ANNUAL_BANDS).count();
  var studyMean = study.select(ANNUAL_BANDS).mean().updateMask(studyCounts.eq(ns));
  var pCount = baseline.select('P').count();
  var siCount = baseline.select('SI').count();
  var baselineMask = pCount.eq(nb).and(siCount.eq(nb));
  var pBaselineMean = baseline.select('P').mean();
  var pBaselineSD = baseline.select('P').reduce(ee.Reducer.sampleStdDev());
  var psi = baseline.select('SI').mean().rename('P_SI').updateMask(baselineMask);
  var nipvMask = pCount.eq(nb).and(pBaselineMean.gt(0));
  var nipv = sampleCV(baseline,'P',nb).rename('P_NIPV');
  var dp = trendOf(study,'P',CFG.studyStart,'dP').updateMask(studyCounts.select('P').eq(ns));
  var dt = trendOf(study,'Ta',CFG.studyStart,'dTa').updateMask(studyCounts.select('Ta').eq(ns));
  var nativePredictors = studyMean.select(['P','Ta','VPD','SM','Rs','P_ET'])
    .addBands(dp).addBands(dt).addBands(psi).addBands(nipv).select(PREDICTORS);
  var target = continuousToTarget(nativePredictors,projection);

  // Integer counts remain nearest-neighbor QA, NOT bilinear fractional years.
  var qStudy = studyCounts.rename(ANNUAL_BANDS.map(function(b){return 'n_years_'+b;}))
    .addBands(studyMonths.select(['negative_P','negative_VPD','negative_Rs','invalid_SM','net_condensation'])
      .sum().rename(['n_negative_P','n_negative_VPD','n_negative_Rs','n_invalid_SM','n_condensation']))
    .updateMask(studyMonths.select('Ta').count().gt(0));
  qStudy = categoricalToTarget(qStudy,projection)
    .addBands(continuousToTarget(studyMean.select('ET').rename('ET_mean_mm_yr'),projection))
    .addBands(target.mask().unmask(0).reduce(ee.Reducer.min()).rename('all_10_predictors_valid'));
  var qBase = baselineMonths.select('P').count().rename('n_valid_months_P')
    .addBands(baselineMonths.select('negative_P').sum().rename('n_negative_P'))
    .addBands(pCount.rename('n_complete_years_P'))
    .addBands(siCount.rename('n_positive_total_years'))
    .addBands(baselineMask.unmask(0).rename('valid_native_SI'))
    .addBands(nipvMask.unmask(0).rename('valid_native_NIPV'))
    .updateMask(baselineMonths.select('negative_P').count().gt(0));
  qBase = categoricalToTarget(qBase,projection)
    .addBands(continuousToTarget(pBaselineMean.rename('baseline_mean_P'),projection))
    .addBands(continuousToTarget(pBaselineSD.rename('baseline_sample_SD_P'),projection));

  var metadataRows = [];
  PREDICTORS.forEach(function(name) {
    var longTerm = name==='P_SI' || name==='P_NIPV';
    var a = longTerm ? CFG.baselineStart : CFG.studyStart;
    var b = longTerm ? CFG.baselineEnd : CFG.studyEnd;
    var metadata = {
      variable:name,units:UNITS[name],first_year:a,last_year:b,dataset:DATASET,
      spatial_processing:'native metric, then bilinear interpolation',
      grid_transform:JSON.stringify(GRID),rows:2800,columns:7200,nodata:NODATA,
      annual_state_average:'equal weights for 12 monthly means',
      precipitation_variability:'mean annual SI; sample CV of annual totals; no detrending',
      VPD_phase:'air temperature selects water or ice for both es and e',
      VPD_temporal_definition:'VPD from monthly mean Ta and Td, then equal-month annual mean; not mean hourly VPD',
      negative_policy:'mask P, VPD, Rs and out-of-range SM; signed ET retained'
    };
    createRasterTask(target.select(name).set(metadata),name,a,b,'bilinear',region);
    metadataRows.push(ee.Feature(null,metadata));
  });
  createRasterTask(qStudy,'QA_study',CFG.studyStart,CFG.studyEnd,'mixed_QA',region);
  createRasterTask(qBase,'QA_baseline',CFG.baselineStart,CFG.baselineEnd,'mixed_QA',region);
  var manifestName = 'ERA5Land_metadata_study_'+CFG.studyStart+'_'+CFG.studyEnd+
    '_baseline_'+CFG.baselineStart+'_'+CFG.baselineEnd+'_bilinear';
  Export.table.toDrive({collection:ee.FeatureCollection(metadataRows),description:manifestName,
    folder:CFG.driveFolder,fileNamePrefix:manifestName,fileFormat:'CSV'});

  if (CFG.exportPointAudit) {
    var coordinates = [[-60,0],[-100,40],[10,50],[25,-10],[80,20],[-77,3.3]];
    var sites = ee.FeatureCollection(coordinates.map(function(xy,i) {
      return ee.Feature(ee.Geometry.Point(xy),{site_id:i,requested_lon:xy[0],requested_lat:xy[1]});
    }));
    var timeList = studyMonths.toList(ns*12);
    var pointRows = ee.FeatureCollection(timeList.map(function(obj) {
      var image = ee.Image(obj);
      return image.select(ANNUAL_BANDS.concat(['Td'])).sampleRegions({
        collection:sites,projection:projection,geometries:true,tileScale:2
      }).map(function(feature) {
        var xy = feature.geometry().coordinates();
        return feature.set({month:image.date().format('YYYY-MM'),
          native_lon:xy.get(0),native_lat:xy.get(1)}).setGeometry(null);
      });
    })).flatten();
    var auditName = 'ERA5Land_native_monthly_six_sites_'+CFG.studyStart+'_'+CFG.studyEnd;
    Export.table.toDrive({collection:pointRows,description:auditName,folder:CFG.driveFolder,
      fileNamePrefix:auditName,fileFormat:'CSV'});
  }
  print('Native projection (read from dataset, never reconstructed)',projection);
  print('Study QA bands',qStudy.bandNames());
  print('Baseline QA bands',qBase.bandNames());
  print('Created 10 predictor TIFs + 2 QA TIFs + metadata CSV; optional native point CSV.',CFG);
  print('Rs is J/m2/day. Divide by 1e6 only when displaying MJ/m2/day.');
  print('QA counts are nearest neighbor; bilinear NIPV need not equal interpolated SD/interpolated mean.');
  print('Start tasks manually; verify 2800 x 7200 dimensions after download.');
  if (CFG.showMaps) {
    Map.setCenter(0,15,2);
    Map.addLayer(target.select('P_SI'),{min:0,max:1.5,palette:['2166ac','f7f7f7','b2182b']},'P.SI');
    Map.addLayer(target.select('VPD'),{min:0,max:30,palette:['2166ac','f7f7f7','b2182b']},'VPD (hPa)',false);
  }
}

function main() {
  validateConfig();
  var first = Math.min(CFG.studyStart,CFG.baselineStart);
  var last = Math.max(CFG.studyEnd,CFG.baselineEnd);
  var raw = ee.ImageCollection(DATASET).filterDate(ee.Date.fromYMD(first,1,1),
    ee.Date.fromYMD(last+1,1,1)).select(SOURCE_BANDS).sort('system:time_start');
  var labels = raw.map(function(image) {
    return image.set('audit_month',image.date().format('YYYY-MM'));
  }).aggregate_array('audit_month');
  var expected = [];
  for (var y=first;y<=last;y++) for(var m=1;m<=12;m++) expected.push(y+'-'+('0'+m).slice(-2));
  labels.evaluate(function(actual,error) {
    if (error || !actual || JSON.stringify(actual.slice().sort())!==JSON.stringify(expected.sort())) {
      print('STOP: missing/duplicate monthly images. No tasks created.',error,actual); return;
    }
    print('PASS: monthly collection complete',actual.length);
    var proceed = function(){buildExports(raw,first,last);};
    if (CFG.formulaChecks) numericalChecks(proceed); else proceed();
  });
}

// EXECUTE: no task is started automatically.
main();
