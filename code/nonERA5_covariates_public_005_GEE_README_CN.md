# 非ERA5解释因子：土壤与地形重算

当前版本：`soil-terrain-small-shards-20260926-3`。针对SOC全局任务的`Reprojection output too large (12377x5437 pixels)`，移除生产数据的强制原始分辨率重投影，改为保留默认原始投影；计算分片由256改为32。公式、缺失值规则和0.05°输出网格不变，云端检查仍开启。

控制台会直接输出`NATIVE_CHECK_JSON`，无需展开FeatureCollection。原始格点与4项数值自检均通过后才创建任务；失败时请查看`CHECK_DETAILS_JSON`。离线测试不能证明此次云端故障已经解决。

主脚本：`nonERA5_covariates_public_005_GEE.js`。已按要求移除全部kNDVI计算、MODIS读取、日期检查、植被QA和年度/16天数据导出。原有数据不删除，ERA5脚本不受影响。

## 运行与输出

在[GEE Code Editor](https://code.earthengine.google.com/)运行。公开资产和波段检查、4项数值自检通过后生成任务，在Tasks中点击Run。当前未实际提交云端计算。

Google Drive文件夹：`JRS_nonERA5_public_005_recomputed`。
默认9项任务：6个指标GeoTIFF、1个覆盖比例QA栅格、1个元数据CSV，另加1个小区域SOC试跑任务。新版输出名带`_v3`，与旧任务区分。

### 此次报错后的操作

1. 在Code Editor使用更新后的完整脚本重新Run，生成新任务；不要直接重试旧的失败任务，其计算过程不会自动更新。
2. 先仅启动`TEST_SOC_boreal_10E_15E_60N_65N_v3`。区域为10–15°E、60–65°N，预期100×100像元，使用与全局完全相同的SOC计算函数。
3. 小区域任务成功后，再启动`SoilGrids_SOC_0_200cm_005deg_area_mean_v3`。已经成功的其他变量不必重复提交。
4. 小区域成功并不保证全球所有位置都通过；若新版仍报相同错误，保留新任务日志后再缩小计算分片或针对投影异常区域拆分，不反复提交同一个失败任务。

`exportShardSize:32`只改变计算分片，不改变分辨率；不是把文件拆成32×32。此错误也不是导出`maxPixels`额度不足，增大`maxPixels`或设置输出`fileDimensions`不能代替对中间计算块的控制。已消耗的21,570.663 EECU-seconds不会因脚本修正被撤回，新任务会另外计量；脚本只创建任务，不自动启动。

| 指标 | 计算 | 单位 |
|---|---|---|
| Clay | SoilGrids六层0–200 cm厚度加权，再聚合至0.05° | g kg⁻¹ |
| Sand | 同上 | g kg⁻¹ |
| SOC | 同上，保留原存储物理单位 | dg kg⁻¹ |
| Nitrogen | 同上，保留原存储物理单位 | cg kg⁻¹ |
| Elevation | GMTED2010_FULL的7.5角秒mea波段聚合均值 | m |
| Slope | 先在上述原始DEM网格计算坡度，再聚合 | ° |

## 保留的处理规则

- 土壤六层厚度权重：`[5,10,15,30,40,100]/200`；每个细像元六层均有效才计算，不将缺失层补0。
- 由细网格到0.05°采用GEE内置的重叠面积加权均值，不重复乘像元面积。
- 目标网格EPSG:4326，`[0.05,0,-180,0,-0.05,80]`，预期2800×7200；下载后核对实际尺寸和像元中心。
- NoData=−9999，聚合完成后才填入，不参与均值。
- QA记录6个指标的有效覆盖比例；`minSpatialCoverage`默认0，对存在有效覆盖的格点取均值。
- 坡度源分辨率为7.5角秒，不是旧表所写30角秒；尚未证明新旧坡度等价，采用前仍需比较并同步修改Table S1。
- Forest_fraction、HFP、WTD和Species仍未替换；不生成森林/非森林标签，不改TAC响应或显著性掩膜。

## 检查

离线测试：`node test_nonERA5_covariates_public_005_GEE.cjs`。
云端自检包括深度加权均值4.95、缺失值保留，以及合成格点聚合均值3与有效覆盖0.75。云端检查与真实数据导出仍需GEE实际运行，离线测试不能替代。

### 自检网格修正

用户曾返回聚合自检值4和覆盖率1，而非预期3和0.75。此时只有资产、波段和深度加权等检查通过，不能认定空间聚合已验证。

自检读取网格与导出CRS、变换参数一致，合成源格点按经纬度定义并显式固定为0.025°。v3在生产聚合中使用原始默认投影，不再强制重投影到原始高分辨率。预期值不变，检查失败仍阻止导出；失败时会额外打印源/目标投影及四个源像元。

这是针对尺度不一致的修正，尚待GEE重新运行确认。`undefined`只是旧日志中的空错误参数，并不意味着资产读取失败。若仍未通过，应根据新增诊断定位，不能关闭自检或将4、1作为预期值。

## 官方依据

- [SoilGrids访问](https://docs.isric.org/globaldata/soilgrids/access_on_gee.html)及[层深与单位](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)
- [GMTED2010_FULL](https://developers.google.com/earth-engine/datasets/catalog/USGS_GMTED2010_FULL)
- [GEE聚合权重](https://developers.google.com/earth-engine/guides/resample#pixel-weights-for-reduceresolution)及[坡度](https://developers.google.com/earth-engine/apidocs/ee-terrain-slope)
- [强制重投影与默认投影区别](https://developers.google.com/earth-engine/apidocs/ee-image-reproject)及[导出shardSize参数](https://developers.google.com/earth-engine/apidocs/export-image-todrive)
