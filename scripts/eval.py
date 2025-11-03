import logging
import os, sys
import numpy as np
import pandas as pd
import datetime
import time

from osgeo import gdal
from utils import setup_logger
from einops import rearrange
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
sys.path.append(os.getcwd())


input_folders = {
    # "Germany-P-mnspi-40": "./work_dir/mnspi/Germany-P/40%/output_temp/images_filled_MNSPI.tif",
    # "Germany-P-mnspi-60": "./work_dir/mnspi/Germany-P/60%/output_temp/images_filled_MNSPI.tif",
    # "Germany-P-mnspi-80": "./work_dir/mnspi/Germany-P/80%/output_temp/images_filled_MNSPI.tif",
        
    # "Germany-P-hants-40": "./work_dir/hants/Germany-P/40%/imputed_images",
    # "Germany-P-hants-60": "./work_dir/hants/Germany-P/60%/imputed_images",
    # "Germany-P-hants-80": "./work_dir/hants/Germany-P/80%/imputed_images",
        
    # "Germany-P-utilise-40": "./work_dir/utilise/Germany-P/40%/imputed_images",
    # "Germany-P-utilise-60": "./work_dir/utilise/Germany-P/60%/imputed_images",
    # "Germany-P-utilise-80": "./work_dir/utilise/Germany-P/80%/imputed_images",
        
    # "Germany-P-nbr3-40":  "./work_dir/Germany-P/others/2025-02-15_T17-50-46_NBRe3-Germany-40%/inference/40%/imputed_images",
    # "Germany-P-nbr3-60":  "./work_dir/Germany-P/others/2025-02-16_T04-55-49_NBRe3-Germany-60%/inference/60%/imputed_images",
    # "Germany-P-nbr3-80":  "./work_dir/Germany-P/others/2025-02-16_T15-37-40_NBRe3-Germany-80%/inference/80%/imputed_images",
        
    # "Germany-P-anytimeformer-40": r"./work_dir/Germany-P/base/2025-02-15_T12-29-57_AnytimeFormer-Germany-40%-r8-128/inference/40%/imputed_images",   
    # "Germany-P-any-anytimeformer-60": r"./work_dir/Germany-P/base/2025-02-15_T12-38-24_AnytimeFormer-Germany-60%-r8-128/inference/60%/imputed_images",
    # "Germany-P-anytimeformer-80": r"./work_dir/Germany-P/base/2025-02-15_T15-38-04_AnytimeFormer-Germany-80%-r8-128/inference/80%/imputed_images",   
    
    # "Germany-mnspi-40": "./work_dir/mnspi/Germany/40%/output_temp/images_filled_MNSPI.tif",
    # "Germany-mnspi-60": "./work_dir/mnspi/Germany/60%/output_temp/images_filled_MNSPI.tif",
    # "Germany-mnspi-80": "./work_dir/mnspi/Germany/80%/output_temp/images_filled_MNSPI.tif",
        
    # "Germany-hants-40": "./work_dir/hants/Germany/40%/imputed_images",
    # "Germany-hants-60": "./work_dir/hants/Germany/60%/imputed_images",
    # "Germany-hants-80": "./work_dir/hants/Germany/80%/imputed_images",
        
    # "Germany-utilise-40": "./work_dir/utilise/Germany/40%/imputed_images",
    # "Germany-utilise-60": "./work_dir/utilise/Germany/60%/imputed_images",
    # "Germany-utilise-80": "./work_dir/utilise/Germany/80%/imputed_images",
        
    # "Germany-nbr3-40":  "./work_dir/Germany/others/2025-01-22_T20-51-25_NBRe3-Germany-40%/inference/40%/imputed_images",
    # "Germany-nbr3-60":  "./work_dir/Germany/others/2025-01-23_T06-29-43_NBRe3-Germany-60%/inference/60%/imputed_images",
    # "Germany-nbr3-80":  "./work_dir/Germany/others/2025-01-23_T17-57-14_NBRe3-Germany-80%/inference/80%/imputed_images",
        
    # "Germany-anytimeformer-40": r"./work_dir/Germany/base/2025-01-17_T14-44-43_AnytimeFormer-Germany-40%-r8-128/inference/40%/imputed_images",   
    # "Germany-anytimeformer-60": r"./work_dir/Germany/base/2025-01-17_T16-42-29_AnytimeFormer-Germany-60%-r8-128/inference/60%/imputed_images",
    # "Germany-anytimeformer-80": r"./work_dir/Germany/base/2025-01-17_T18-39-32_AnytimeFormer-Germany-80%-r8-128/inference/80%/imputed_images",   
    
    # #######################
    # "Hebei-mnspi-40": "./work_dir/mnspi/Hebei/40%/output_temp/images_filled_MNSPI.tif",
    # "Hebei-mnspi-60": "./work_dir/mnspi/Hebei/60%/output_temp/images_filled_MNSPI.tif",
    # "Hebei-mnspi-80": "./work_dir/mnspi/Hebei/80%/output_temp/images_filled_MNSPI.tif",
        
    # "Hebei-hants-40": "./work_dir/hants/Hebei/40%/imputed_images",
    # "Hebei-hants-60": "./work_dir/hants/Hebei/60%/imputed_images",
    # "Hebei-hants-80": "./work_dir/hants/Hebei/80%/imputed_images",
        
    # "Hebei-utilise-40": "./work_dir/utilise/Hebei/40%/imputed_images",
    # "Hebei-utilise-60": "./work_dir/utilise/Hebei/60%/imputed_images",
    # "Hebei-utilise-80": "./work_dir/utilise/Hebei/80%/imputed_images",
    
    # "Hebei-nbr3-40": "./work_dir/Hebei/others/2025-01-23_T09-24-12_NBRe3-Hebei-40%/inference/40%/imputed_images",
    # "Hebei-nbr3-60": "./work_dir/Hebei/others/2025-01-23_T22-06-47_NBRe3-Hebei-60%/inference/60%/imputed_images",
    # "Hebei-nbr3-80": "./work_dir/Hebei/others/2025-01-24_T10-22-22_NBRe3-Hebei-80%/inference/80%/imputed_images",
        
    # "Hebei-anytimeformer-40": r"./work_dir/Hebei/base/2025-01-17_T16-56-23_AnytimeFormer-Hebei-40%-r8-128/inference/40%/imputed_images",
    # "Hebei-anytimeformer-60": r"./work_dir/Hebei/base/2025-01-17_T14-53-17_AnytimeFormer-Hebei-60%-r8-128/inference/60%/imputed_images",
    # "Hebei-anytimeformer-80": r"./work_dir/Hebei/base/2025-01-17_T18-59-34_AnytimeFormer-Hebei-80%-r8-128/inference/80%/imputed_images",
    
    # ##########################  
    # "California-mnspi-40": "./work_dir/mnspi/California/40%/output_temp/images_filled_MNSPI.tif",
    # "California-mnspi-60": "./work_dir/mnspi/California/60%/output_temp/images_filled_MNSPI.tif",
    # "California-mnspi-80": "./work_dir/mnspi/California/80%/output_temp/images_filled_MNSPI.tif",
        
    # "California-hants-40": "./work_dir/hants/California/40%/imputed_images",
    # "California-hants-60": "./work_dir/hants/California/60%/imputed_images",
    # "California-hants-80": "./work_dir/hants/California/80%/imputed_images",
        
    # "California-utilise-40": "./work_dir/utilise/California/40%/imputed_images",
    # "California-utilise-60": "./work_dir/utilise/California/60%/imputed_images",
    # "California-utilise-80": "./work_dir/utilise/California/80%/imputed_images",
        
    # "California-nbr3-40": "./work_dir/CA/others/2025-01-21_T11-41-17_NBRe3-CA-40%/inference/40%/imputed_images",
    # "California-nbr3-60": "./work_dir/CA/others/2025-01-21_T23-17-13_NBRe3-CA-60%/inference/60%/imputed_images",
    # "California-nbr3-80": "./work_dir/CA/others/2025-01-22_T10-50-54_NBRe3-CA-80%/inference/80%/imputed_images",
        
    # "California-anytimeformer-40": r"./work_dir/CA/base/2025-01-17_T14-43-54_AnytimeFormer-CA-40%-r8-128/inference/40%/imputed_images",
    # "California-anytimeformer-60": r"./work_dir/CA/base/2025-01-17_T16-42-02_AnytimeFormer-CA-60%-r8-128/inference/60%/imputed_images",
    # "California-anytimeformer-80": r"./work_dir/CA/base/2025-01-17_T18-41-30_AnytimeFormer-CA-80%-r8-128/inference/80%/imputed_images",
    
    ##########################
    "Ken-nbr3-40": "./work_dir/Hebei/others/2025-01-23_T09-24-12_NBRe3-Hebei-40%/inference/40%/imputed_images",
    "Ken-nbr3-60": "./work_dir/Hebei/others/2025-01-23_T22-06-47_NBRe3-Hebei-60%/inference/60%/imputed_images",
    "Ken-nbr3-80": "./work_dir/Hebei/others/2025-01-24_T10-22-22_NBRe3-Hebei-80%/inference/80%/imputed_images",
        
    "Ken-anytimeformer-40": r"./work_dir/Hebei/base/2025-01-17_T16-56-23_AnytimeFormer-Hebei-40%-r8-128/inference/40%/imputed_images",
    "Ken-anytimeformer-60": r"./work_dir/Hebei/base/2025-01-17_T14-53-17_AnytimeFormer-Hebei-60%-r8-128/inference/60%/imputed_images",
    "Ken-anytimeformer-80": r"./work_dir/Hebei/base/2025-01-17_T18-59-34_AnytimeFormer-Hebei-80%-r8-128/inference/80%/imputed_images",    
    
    "Nev-nbr3-40": "./work_dir/Hebei/others/2025-01-23_T09-24-12_NBRe3-Hebei-40%/inference/40%/imputed_images",
    "Nev-nbr3-60": "./work_dir/Hebei/others/2025-01-23_T22-06-47_NBRe3-Hebei-60%/inference/60%/imputed_images",
    "Nev-nbr3-80": "./work_dir/Hebei/others/2025-01-24_T10-22-22_NBRe3-Hebei-80%/inference/80%/imputed_images",
        
    "Nev-anytimeformer-40": r"./work_dir/Hebei/base/2025-01-17_T16-56-23_AnytimeFormer-Hebei-40%-r8-128/inference/40%/imputed_images",
    "Nev-anytimeformer-60": r"./work_dir/Hebei/base/2025-01-17_T14-53-17_AnytimeFormer-Hebei-60%-r8-128/inference/60%/imputed_images",
    "Nev-anytimeformer-80": r"./work_dir/Hebei/base/2025-01-17_T18-59-34_AnytimeFormer-Hebei-80%-r8-128/inference/80%/imputed_images",    
    
    "Ind-nbr3-80": r"./work_dir/Hebei/base/2025-01-17_T18-59-34_AnytimeFormer-Hebei-80%-r8-128/inference/80%/imputed_images", 
    "Ind-anytimeformer-80": r"./work_dir/Hebei/base/2025-01-17_T18-59-34_AnytimeFormer-Hebei-80%-r8-128/inference/80%/imputed_images",   
}


truth_tif_folders = {
    # "Germany": "./dataset/data_truth/Germany/tif",
    # "Hebei": "./dataset/data_truth/Hebei/tif",
    # "California": "./dataset/data_truth/California/tif",
    # "Germany-P": "./dataset/data_truth/Germany-P/tif",
    
    "Ken": "./dataset/data_truth/Ken/tif",
    "Nev": "./dataset/data_truth/Nev/tif",
    "Ind": "./dataset/data_truth/Ind/tif",
}


class MetricNumpy:
    def __init__(self, scale):
        self.scale = scale
    
    def masked_mae_cal(self, inputs, target, mask):
        """calculate Mean Absolute Error"""
        inputs = (inputs / self.scale).astype(np.float32)
        target = (target / self.scale).astype(np.float32)
        error_sum = np.sum(np.abs(inputs - target) * mask)
        num = np.sum(mask) + 1e-9
        return error_sum / num

    def _masked_mse_cal(self, inputs, target, mask):
        """calculate Mean Square Error"""
        return np.sum(np.square(inputs - target) * mask) / (np.sum(mask) + 1e-9)

    def masked_rmse_cal(self, inputs, target, mask):
        """calculate Root Mean Square Error"""
        inputs = (inputs / self.scale).astype(np.float32)
        target = (target / self.scale).astype(np.float32)
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
        inputs = (inputs / self.scale).astype(np.float32)
        target = (target / self.scale).astype(np.float32)
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
    

def read_tif_files(folder):
    tif_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
    tif_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
    return tif_files


def stack_tif_files(tif_files):
    arrays = []
    for tif in tif_files:
        ds = gdal.Open(tif)
        arrays.append(ds.ReadAsArray())
    return np.stack(arrays, axis=0)


def main(missing_tif_folder, pred_tif_folder, truth_tif_folder, logger, only_ssim_psnr=False):
    metric_computer = MetricNumpy(scale=1)

    missing_stack = stack_tif_files(read_tif_files(missing_tif_folder))
    truth_stack = stack_tif_files(read_tif_files(truth_tif_folder)) / 10000.0
    t, b, _, _ = truth_stack.shape
    
    if os.path.isdir(pred_tif_folder):
        input_tif_files = read_tif_files(pred_tif_folder)
        input_stack = stack_tif_files(input_tif_files) / 10000.0
        
    elif os.path.isfile(pred_tif_folder) and pred_tif_folder.endswith('.tif'):
        input_stack_temp = gdal.Open(pred_tif_folder).ReadAsArray()
        input_stack_temp /= 10000.0
        input_stack = np.zeros_like(truth_stack)
        for i in range(t):
            input_stack[i, :, :, :] = input_stack_temp[i*b:(i+1)*b, :, :]
        del input_stack_temp
    
    # 根据保留验证的mask计算验证指标，验证mask：missing_stack为0，但truth不为0的地方为1
    mask = np.logical_and(missing_stack == 0, truth_stack != 0)
    logger.info("Performance metrics for {} are listed as follows:".format(pred_tif_folder))
    
    metric_dict = {}
    if not only_ssim_psnr:
        # cal MAE, R2, RMSE ...
        metric_dict = {
            "MAE": metric_computer.masked_mae_cal(input_stack, truth_stack, mask),
            "RMSE": metric_computer.masked_rmse_cal(input_stack, truth_stack, mask),
            "R2": metric_computer.masked_r2_score(input_stack, truth_stack, mask),
        }
        logger.info("Performance metrics are listed as follows:")
        for k, v in metric_dict.items():
            logger.info(f"{k}: {np.round(v, 4)}")  
    
    # 计算SSIM和PSNR
    mask = mask[:, 0, :, :]
    cal_indictors = np.sum(mask, axis=(1, 2)) > 0
    psnr = []
    ssim = []
    for i, f in enumerate(cal_indictors):
        if not f:
            continue
        for j in range(b):
            true_img = truth_stack[i, j, :, :].squeeze()
            pred_img = input_stack[i, j, :, :].squeeze()
            
            pred_img[true_img==0] = 0
            
            psnr_temp = peak_signal_noise_ratio(true_img, pred_img, data_range=1)
            ssim_temp = structural_similarity(true_img, pred_img, data_range=1)
            psnr.append(psnr_temp)
            ssim.append(ssim_temp)
        
    psnr = np.mean(psnr)
    ssim = np.mean(ssim)
    metric_dict["SSIM"] = ssim
    metric_dict["PSNR"] = psnr
    
    logger.info(f"PSNR: {np.round(psnr, 3)}")
    logger.info(f"SSIM: {np.round(ssim, 3)}")
    return metric_dict
        
    
if __name__ == '__main__':
    
    logger = setup_logger("psnr_ssim_metric_mod-ken-nev-ind.log", "metric") 
    only_ssim_psnr = False
    
    df = pd.DataFrame(columns=["Site", "Ratio", "Method", "R2", "RMSE", "MAE", "SSIM", "PSNR"])
    
    for k, pred_tif_folder in tqdm(input_folders.items()):
        site = k.split('-')[0] # Check !!!!!
        ratio = k.split('-')[-1]
        method = k.split('-')[1]
        
        truth_tif_folder = truth_tif_folders[site]
        missing_tif_folder = f"./dataset/data_for_model/{site}/tif/{ratio}" + "%"
                
        try:
            metric_dict = main(
                missing_tif_folder,
                pred_tif_folder,
                truth_tif_folder,
                logger,
                only_ssim_psnr=only_ssim_psnr
            )
            new_row = pd.DataFrame([{
                "Site": site,
                "Ratio": ratio,
                "Method": method,
                "R2": metric_dict.get("R2", None),
                "RMSE": metric_dict.get("RMSE", None),
                "MAE": metric_dict.get("MAE", None),
                "SSIM": metric_dict.get("SSIM", None),
                "PSNR": metric_dict.get("PSNR", None)
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        except:
            logger.info(f"Error in {k}")
            continue
        
    df.to_csv("metrics-Ken-Nev-Ind.csv", index=False)