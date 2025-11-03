# 数据准备指南 ([English](README.md))

本文档提供了项目所用数据集的详细准备步骤，包括通过 **Google Earth Engine (GEE)** 下载数据和本地预处理的说明。

---

## 1. 从 GEE 下载数据

### 1.1 Sentinel-2 数据

**GEE 脚本：**  
[https://code.earthengine.google.com/2626584d1084dea06f89488fac038b12](https://code.earthengine.google.com/2626584d1084dea06f89488fac038b12)

**需要修改的参数：**

| 参数 | 描述 | 示例 |
|------|------|------|
| `var table` | 研究区域shpfile，可从 [HuggingFace](https://huggingface.co/datasets/tangkaii/AnytimeFormer/tree/main) 下载论文研究区 | — |
| `var start` | 起始日期 | `ee.Date('2019-01-01')` |
| `var end` | 结束日期 | `ee.Date('2019-01-31')` |
| `var crstype` | EPSG 投影编码 | `'EPSG:32633'` |
| `var folder` | 目标 Google Drive 文件夹 | `'S2-multibands'` |

> **注意：** 由于 GEE 内存限制，建议每次脚本运行仅下载 3–4 个月的数据。

---

### 1.2 Sentinel-1 数据

**GEE 脚本：**  
[https://code.earthengine.google.com/e4f1735d3f995752ec3e0eb7112a8654](https://code.earthengine.google.com/e4f1735d3f995752ec3e0eb7112a8654)

**需要修改的参数：**

| 参数 | 描述 | 示例 |
|------|------|------|
| `table (study area shp)` | 研究区域的 Shapefile 文件 | — |
| `crs` | EPSG 投影编码 | `'EPSG:32633'` |
| `START_DATE` | 起始日期 | `"2019-01-01"` |
| `STOP_DATE` | 结束日期 | `"2019-01-31"` |
| `ORBIT` | 轨道方向（`BOTH`、`ASCENDING` 或 `DESCENDING`） | `'ASCENDING'` |

**注意事项：**
- 建议仅使用单一轨道（`ASCENDING` 或 `DESCENDING`）。  
- 某些区域可能缺少某一轨道的数据，请根据检索结果调整。  
- 与 Sentinel-2 相同，建议每次仅下载 3–4 个月的数据。

---

### 1.3 可选项：云相关数据

如果基于 Sentinel-2 的 SCL 云去除效果良好，则可跳过下载 Cloud Score/Probability 数据集。

- **Cloud Probability（云概率）：** [https://code.earthengine.google.com/0f06874e01d7738bd2f8ddcf5c3db798](https://code.earthengine.google.com/0f06874e01d7738bd2f8ddcf5c3db798)  
- **Cloud Score（云得分）：** [https://code.earthengine.google.com/9f4aaaa8b7a075a9d07aad98b0191e44](https://code.earthengine.google.com/9f4aaaa8b7a075a9d07aad98b0191e44)

**需要修改的参数：**  
与 Sentinel-1/2 相同，包括研究区 shpfile、时间范围、目标文件夹等。

---

## 2. 本地云去除与数据集生成

### 2.1 文件夹结构

请将本地数据组织如下：

```
dataset  
|-- dataset_down_from_GEE  
|   |-- Germany  
|   |   |-- S1_raw             # GEE下载S1数据解压
|   |   |-- S2_raw             # GEE下载S2数据解压，未去云
|   |   |-- S2_remove_cloud    # 运行去云脚本后生成  
|   |   |-- Cloud_Probability  # GEE下载的云概率
|   |   |-- Cloud_Score        # GEE下载的云分数
|   |   |-- cand_cloud_mask    # 用于人工加云模拟的mask影像
|-- dataset_for_model  
|   |-- Germany             
|   |   |-- tif                # 用于获取参考投影的原始S2 TIF文件
|   |   |   |-- 40%            # 40%缺失比例下的TIF时间序列影像 （单景存储如 S2_L2A_20190208.tif....）
|   |   |   |-- 60%            # 60%缺失比例下的TIF时间序列影像
|   |   |   |-- 80%            # 80%缺失比例下的TIF时间序列影像
|   |   |-- hdf                # 模型训练和推理
```

> 请确保文件夹命名与您的 **研究区域** 和 **处理脚本** 保持一致。

---

### 2.2 预处理流程

#### 步骤 1. 云去除

运行位于 `preprocess` 文件夹中的脚本：

```bash
python 1_remove_cloud.py
```
#### 步骤 2. 独立验证数据生成（目的为应用可跳过）
该模式将通过人工施加云mask来模拟不同比例缺失的数据，其中被mask掉的pixels作为独立验证集：

```bash
python 2_pre_cand_cloud_masks.py
```

```bash
jupyter notebook 3_generate_hdf_for_Germany.ipynb
```

这将生成以下文件：
```
dataset/dataset_for_model/<site>/hdf/random_missing.hdf
```

#### 步骤 3. Anytime模式

若目的为应用Anytime模式，请运行：
```bash
jupyter notebook 3_generate_hdf_for_Germany_anytime.ipynb
```

输出文件路径：
这将生成以下文件：
```
dataset/dataset_for_model/<site>/hdf/anytime.hdf
```

### 2.3 其他说明

- 请确保脚本中的 **所有路径** 均正确对应您的本地环境。  
- 云去除与数据准备的耗时与 **研究区大小** 和 **时间范围** 有关。  
- 建议在生成 HDF 数据集前，先检查 **中间结果**（尤其是 `S2_remove_cloud` 文件夹）。  
- 为提高工作效率：  
  - 从 GEE 导出时建议使用 **≤ 4 个月** 的时间窗口；  
  - 确保 **Google Drive 存储空间充足** 以保存中间 `.tif` 文件；  
  - 保证所有 **Sentinel-1** 与 **Sentinel-2** 图层的 **CRS 投影一致**。  
