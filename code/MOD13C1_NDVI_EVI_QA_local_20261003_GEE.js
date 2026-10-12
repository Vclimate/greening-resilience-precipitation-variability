// Native 16-day NDVI/EVI and QA for LOCAL TAC calculation, 2000-2025.
// No extra scaling. No annual average. No TAC calculation. No resampling.
// QA is retained as-is; the local analysis must use SummaryQA == 0.
// Python submission already creates these tasks. Keep false to avoid duplicates.
var CREATE_TASKS = false;
var SOURCE = 'MODIS/061/MOD13C1';
var FOLDER = 'JRS_MOD13C1_NDVI_EVI_QA_16day_005deg_20261003';
var TRANSFORM = [0.05, 0, -180, 0, -0.05, 80];
var BANDS = ['NDVI', 'EVI', 'SummaryQA'];
var ROI = ee.Geometry.Rectangle([-180+1e-8, -60+1e-8, 180-1e-8, 80-1e-8], null, false);
var raw = ee.ImageCollection(SOURCE).filterDate('2000-01-01', '2026-01-01')
  .sort('system:time_start');
print('Source projection', raw.first().select('NDVI').projection());
print('Dates available (2000 starts in February)', raw.size());

var metadata = raw.map(function(image) {
  return image.set('export_date', ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'));
});
ee.Dictionary({dates: metadata.aggregate_array('export_date'),
  indexes: metadata.aggregate_array('system:index')}).evaluate(function(info, error) {
  if (error) { print('STOP: metadata query failed', error); return; }
  if (!CREATE_TASKS) { print('Preview only; tasks already submitted with Python.'); return; }
  for (var year = 2000; year <= 2025; year++) {
    for (var half = 1; half <= 2; half++) {
      var images = [], labels = [];
      info.dates.forEach(function(date, i) {
        var month = Number(date.slice(5, 7));
        if (Number(date.slice(0, 4)) !== year || (month <= 6 ? 1 : 2) !== half) return;
        var stamp = date.split('-').join('');
        var names = BANDS.map(function(band) { return band + '_' + stamp; });
        var image = ee.Image(SOURCE + '/' + info.indexes[i]).select(BANDS)
          .toFloat().rename(names).unmask(-9999, false);
        images.push(image);
        labels = labels.concat(names);
      });
      if (!images.length || images.length > 12) throw new Error('Unexpected image count: ' + year + ' H' + half);
      var stack = ee.ImageCollection.fromImages(images).toBands().rename(labels);
      var name = 'MOD13C1_NDVI_EVI_QA_' + year + '_H' + half + '_raw_20261003';
      Export.image.toDrive({image: stack, description: name, folder: FOLDER,
        fileNamePrefix: name, region: ROI, crs: 'EPSG:4326', crsTransform: TRANSFORM,
        maxPixels: 1e13, fileFormat: 'GeoTIFF',
        formatOptions: {cloudOptimized: true, noData: -9999}});
    }
  }
  print('52 half-year export tasks created. Each band retains one source date.');
});
