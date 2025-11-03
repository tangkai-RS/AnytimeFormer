import os
import time
import torch
import logging
import random
import datetime
import numpy as np

from functools import reduce
from osgeo import gdal


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    error_sum = torch.sum(torch.abs(inputs - target) * mask)
    num = torch.sum(mask) + 1e-9
    return error_sum / num


def masked_mse_cal(inputs, target, mask):
    """calculate Mean Square Error"""
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """calculate Root Mean Square Error"""
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """calculate Mean Relative Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (
        torch.sum(torch.abs(target * mask)) + 1e-9
    )


def masked_mape_cal(y_pred, y_true, mask):
    """Mean Absolute Percentage Error (MAPE)"""
    absolute_percentage_error = torch.abs((y_true - y_pred) / y_true)
    masked_absolute_percentage_error = absolute_percentage_error * mask
    valid_elements = mask.sum() + 1e-9
    mape = (masked_absolute_percentage_error.sum() / valid_elements) * 100.
    return mape


def total_variation_loss(inputs):
    """calculate total_variation_loss for smooth time-series"""
    diff = inputs[:, 1:] - inputs[:, :-1]
    loss = torch.sum(torch.abs(diff)) / diff.numel()
    return loss


def setup_logger(log_file_path, log_name, mode="a"):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def load_model(model, saved_model_path, logger):
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Already restored model from checkpoint: {saved_model_path}")
    return model


def count_model_parameters(module):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    print('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))
    return cnt  


def measure_inference_speed(model, data, bs=1024, max_iter=200, log_interval=50):
    '''
    data includes 2048 samples
    '''
    model.eval()
    
    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0
 
    # benchmark with 65536 samples and take the average
    for i in range(max_iter):
 
        torch.cuda.synchronize()
        start_time = time.perf_counter()
 
        with torch.no_grad():
            model(data)
 
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
 
        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done samples [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} * {bs:.0f} samples / s, '
                    f'times per ({bs:.0f} samples): {1000 / fps:.1f} ms / bs',
                    flush=True)
 
        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per ({bs:.0f} samples): {1000 / fps:.1f} ms / bs',
                flush=True)
            break
    return fps


class MetricNumpy:
    def __init__(self, args):
        # convert int16 to float32
        self.scale = args.scale
    
    def masked_mae_cal(self, inputs, target, mask):
        """calculate Mean Absolute Error"""
        # inputs = (inputs / self.scale).astype(np.float32)
        # target = (target / self.scale).astype(np.float32)
        error_sum = np.sum(np.abs(inputs - target) * mask)
        num = np.sum(mask) + 1e-9
        return error_sum / num

    def _masked_mse_cal(self, inputs, target, mask):
        """calculate Mean Square Error"""
        return np.sum(np.square(inputs - target) * mask) / (np.sum(mask) + 1e-9)

    def masked_rmse_cal(self, inputs, target, mask):
        """calculate Root Mean Square Error"""
        # inputs = (inputs / self.scale).astype(np.float32)
        # target = (target / self.scale).astype(np.float32)
        return np.sqrt(self._masked_mse_cal(inputs, target, mask))    

    def masked_correlation_cal(self, inputs, target, mask):
        """calculate overall correlation coefficient"""
        inputs = (inputs / self.scale).astype(np.float32)
        target = (target / self.scale).astype(np.float32)
        # Apply mask
        inputs = inputs * mask
        target = target * mask

        # Flatten the arrays to calculate overall correlation
        inputs = inputs.flatten()
        target = target.flatten()
        mask = mask.flatten()

        # Apply mask
        inputs = inputs[mask == 1]
        target = target[mask == 1]

        # Calculate mean
        input_mean = np.mean(inputs)
        target_mean = np.mean(target)

        # Calculate covariance and variances
        covariance = np.sum((inputs - input_mean) * (target - target_mean))
        input_variance = np.sum((inputs - input_mean) ** 2)
        target_variance = np.sum((target - target_mean) ** 2)

        # Calculate correlation coefficient
        correlation = covariance / (np.sqrt(input_variance) * np.sqrt(target_variance) + 1e-9)

        return correlation

    def masked_r2_score(self, inputs, target, mask):
        """calculate R^2 (coefficient of determination)"""
        # inputs = (inputs / self.scale).astype(np.float32)
        # target = (target / self.scale).astype(np.float32)
        # Apply mask
        inputs = inputs * mask
        target = target * mask

        # Flatten the arrays to calculate overall R^2
        inputs = inputs.flatten()
        target = target.flatten()
        mask = mask.flatten()

        # Apply mask
        inputs = inputs[mask == 1]
        target = target[mask == 1]

        # Calculate mean of target
        target_mean = np.mean(target)

        # Calculate total sum of squares (SST) and residual sum of squares (SSR)
        sst = np.sum((target - target_mean) ** 2)
        ssr = np.sum((target - inputs) ** 2)

        # Calculate R^2
        r2 = 1 - (ssr / (sst + 1e-9))

        return r2
    
    def masked_mape(self, y_pred, y_true, mask):
        absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
        masked_absolute_percentage_error = absolute_percentage_error * mask
        valid_elements = mask.sum() + 1e-9
        mape = (masked_absolute_percentage_error.sum() / valid_elements) * 100.  # 转换为百分比
        return mape
    

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
    elif dtype == 'int16':
        datatype = gdal.GDT_Int16
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
    

def doy_to_ymd(year=2019, doy=1):
    date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(doy) - 1)
    return date.strftime('%Y-%m-%d')
 

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        

def doy_of_year(year):
    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)    
    
    if is_leap_year(year):
        date_output = np.arange(1, 367)
    else:
        date_output = np.arange(1, 366)
    return date_output


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")