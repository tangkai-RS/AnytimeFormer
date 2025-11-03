import datetime
import os
import numpy as np
from osgeo import gdal



def imread(tif_file):
    return gdal.Open(tif_file).ReadAsArray()


def imsave(img, path, dtype=None, ref_img_path=None, no_data=None, gtf=None):
    if len(img.shape) == 3:
        (n, h, w) = img.shape
    else:
        (h, w) = img.shape
        n = 1
       
    if dtype == 'uint8':
        datatype = gdal.GDT_Byte
    elif dtype == 'uint16':
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, w, h, n, datatype, ['COMPRESS=LZW', 'BIGTIFF=YES'])   
    
    if ref_img_path is not None:
        dataset_ref = gdal.Open(ref_img_path)  
        datatype = dataset_ref.GetRasterBand(1).DataType    # Subject to the first band
        proj = dataset_ref.GetProjection()
        if gtf is None:
            gtf = dataset_ref.GetGeoTransform()
        # gtf_new = (gtf[0], gtf[1], gtf[2],gtf[3]+10,gtf[4], gtf[5])
        dataset.SetProjection(proj)
        dataset.SetGeoTransform(gtf)
            
    if len(img.shape) == 3:
        for t in range(n):
            dataset.GetRasterBand(t + 1).WriteArray(img[t])
            if no_data is not None:
                dataset.GetRasterBand(t + 1).SetNoDataValue(no_data)
    else:
        dataset.GetRasterBand(1).WriteArray(img)
        if no_data is not None:
            dataset.GetRasterBand(1).SetNoDataValue(no_data)
    del dataset
    

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def doy_to_ymd(year, doy):
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    return date.strftime('%Y-%m-%d')