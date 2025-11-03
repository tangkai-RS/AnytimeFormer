# Data Preparation Guide ([中文](README_ch.md))

This document provides detailed instructions for preparing the dataset used in this project, including data download via **Google Earth Engine (GEE)** and local preprocessing.

---

## 1. Downloading Data from GEE

### 1.1 Sentinel-2 Data

**GEE Script:**  
[https://code.earthengine.google.com/2626584d1084dea06f89488fac038b12](https://code.earthengine.google.com/2626584d1084dea06f89488fac038b12)

**Parameters to modify:**

| Parameter | Description | Example |
|------------|--------------|----------|
| `var table` | study area shpfile, downloadable from [HuggingFace](https://huggingface.co/datasets/tangkaii/AnytimeFormer/tree/main) | — |
| `var start` | Start date | `ee.Date('2019-01-01')` |
| `var end` | End date | `ee.Date('2019-01-31')` |
| `var crstype` | EPSG projection code | `'EPSG:32633'` |
| `var folder` | Target folder on Google Drive | `'S2-multibands'` |

> **Note:** Due to GEE memory limitations, it is recommended to download 3–4 months of data per script execution.

---

### 1.2 Sentinel-1 Data

**GEE Script:**  
[https://code.earthengine.google.com/e4f1735d3f995752ec3e0eb7112a8654](https://code.earthengine.google.com/e4f1735d3f995752ec3e0eb7112a8654)

**Parameters to modify:**

| Parameter | Description | Example |
|------------|--------------|----------|
| `table (study area shp)` | Shapefile of your study area | — |
| `crs` | EPSG projection code | `'EPSG:32633'` |
| `START_DATE` | Start date | `"2019-01-01"` |
| `STOP_DATE` | End date | `"2019-01-31"` |
| `ORBIT` | Orbit direction (`BOTH`, `ASCENDING`, or `DESCENDING`) | `'ASCENDING'` |

**Notes:**
- Use only a single orbit (`ASCENDING` or `DESCENDING`).  
- Some regions may lack data for one orbit; adjust based on your search results.  
- As with Sentinel-2, downloading 3–4 months of data per run is recommended.

---

### 1.3 Optional: Cloud-Related Data

If Sentinel-2’s SCL-based cloud removal works well, you can skip downloading Cloud Score/Probability datasets.

- **Cloud Probability:** [https://code.earthengine.google.com/0f06874e01d7738bd2f8ddcf5c3db798](https://code.earthengine.google.com/0f06874e01d7738bd2f8ddcf5c3db798)  
- **Cloud Score:** [https://code.earthengine.google.com/9f4aaaa8b7a075a9d07aad98b0191e44](https://code.earthengine.google.com/9f4aaaa8b7a075a9d07aad98b0191e44)

**Parameters to modify:**
Same as Sentinel-1/2 — study area shpfile, date range, target folder, etc.

---

## 2. Local Cloud Removal and Dataset Generation

### 2.1 Folder Structure

Organize your local data as follows:

```
dataset  
├── dataset_down_from_GEE  
│   └── Germany  
│       ├── S1_raw             # Extracted Sentinel-1 data from GEE  
│       ├── S2_raw             # Extracted Sentinel-2 data from GEE (before cloud removal)  
│       ├── S2_remove_cloud    # Generated after running the cloud removal script  
│       ├── Cloud_Probability  # Cloud probability data downloaded from GEE  
│       ├── Cloud_Score        # Cloud score data downloaded from GEE  
│       └── cand_cloud_mask    # Mask images used for artificial cloud simulation  
└── dataset_for_model  
    └── Germany             
        ├── tif                # Original Sentinel-2 TIF files used to obtain reference projection  
        │   ├── 40%            # Time series with 40% missing ratio (stored individually, e.g., S2_L2A_20190208.tif ...)  
        │   ├── 60%            # Time series with 60% missing ratio  
        │   └── 80%            # Time series with 80% missing ratio  
        └── hdf                # For model training and inference  
```

> Ensure that folder names match your **study area** and **processing scripts**.

---

### 2.2 Preprocessing Workflow

#### Step 1. Cloud Removal

Run the following script located in the `preprocess` folder:

```bash
python 1_remove_cloud.py
```
#### Step 2. Normal mode with independent test pixels (this step can be skipped in the application) 

This mode will simulate data missing at different rates by artificially applying cloud masks, with the masked pixels serving as an independent validation set, run:

```bash
python 2_pre_cand_cloud_masks.py
```
```bash
jupyter notebook 3_generate_hdf_for_Germany-P.ipynb
```

This will create the following file:
```
dataset/dataset_for_model/<site>/hdf/random_missing.hdf
```

#### Step 3. Anytime Mode
For direct model application or inference, run:

```bash
jupyter notebook 3_generate_hdf_for_Germany-P_anytime.ipynb
```

dataset/dataset_for_model/<site>/hdf/anytime.hdf

### 2.3 Additional Notes

- Make sure that **all file paths** inside the scripts are correctly configured for your environment.  
- Cloud removal and data preparation can take a long time depending on your **region size** and **date range**.  
- It is recommended to check **intermediate outputs** (especially `S2_remove_cloud`) before generating HDF datasets.  
- For efficient workflow:
  - Use **smaller temporal windows (≤ 4 months)** when exporting from GEE.  
  - Ensure sufficient **Google Drive storage** for intermediate `.tif` exports.  
  - Validate **CRS consistency** across all Sentinel-1 and Sentinel-2 layers.  
