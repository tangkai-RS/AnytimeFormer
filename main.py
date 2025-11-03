import argparse
import shutil
import os, sys
import yaml
import h5py
import numpy as np
import torch

from einops import rearrange
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from model import model_dict
from model.utils import MetricNumpy
from model.utils import (
    setup_logger, seed_torch, load_model, str2bool,
    check_path, doy_to_ymd, imread, imsave, doy_of_year
)


class MyDataset(Dataset):
    """Dataset for loading train/test samples with optional auxiliary data (SAR)."""
    
    def __init__(
        self, file_path, 
        split="train", # train or test
        args=None
    ):
        """
        Args:
            file_path (str): HDF5 file path containing the dataset.
            split (str): Dataset split mode, "train" or "test".
            args: Namespace or config containing additional settings (scale, with_X_aux, gap_mode, etc.).
        """
        self.args = args
        self.artificial_missing_rate = args.artificial_missing_rate
        self.split = split
        self.scale = args.scale # convert int16 to float
        self.with_X_aux = args.with_X_aux
        self.file_path = file_path
        self.ratio = args.ratio
        self.gap_mode = args.gap_mode
            
        with h5py.File(self.file_path, "r") as hf:
            self.X = hf["data"]["X"][:] # 0 is missing
            self.date_output = hf["data"]["date"][:] # output date
            self._prepare_dataset_names()
            
            self.X_mean = hf["data"][self.X_mean_name][:]
            self.X_std = hf["data"][self.X_std_name][:]
            self.date_train = hf["data"][self.date_train_name][:]
            self.date_aux = np.asarray(0.)
            self.missing_mask = hf["mask"][self.miss_mask_name][:] # 0 is missing 
            
            if self.with_X_aux:
                self.X_aux = hf["data"]["X_aux"][:] # SAR data
                self.date_aux = hf["data"]["date_aux"][:] # SAR date
                self.X_aux_mean = hf["data"]["X_aux_mean"][:]
                self.X_aux_std = hf["data"]["X_aux_std"][:]
                self.X_aux = np.nan_to_num(self.X_aux)
                self.X_aux = self._standscaler_Xaux(self.X_aux)
                
            if self.split == "test":
                self.indicating_mask = hf["mask"][self.indicating_mask_name][:] # 1 is missing
        # standardize X
        self.X = self._standscaler_X(self.X)
        
        # for debugging purposes
        if self.split == "train" and args.debug_mode:
            self.X = self.X[0:1000, ...]
            self.X_aux = self.X_aux[0:1000, ...]
        if self.gap_mode == "continuous":
            self.date_train = self.date_output.copy()
               
    def __len__(self):
        return len(self.X)
    
    def _prepare_dataset_names(self):
        if self.gap_mode == "random":
            self.miss_mask_name = f"missing_mask_{self.ratio}"
            self.indicating_mask_name = f"indicating_mask_{self.ratio}"
            self.X_mean_name = f"X_mean_{self.ratio}"
            self.X_std_name = f"X_std_{self.ratio}"
            self.date_train_name = "date"
        elif self.gap_mode == "continuous":
            self.miss_mask_name = f"missing_mask_{self.args.gap_doy}"
            self.indicating_mask_name = f"indicating_mask_{self.args.gap_doy}"
            self.X_mean_name = "X_mean"
            self.X_std_name = "X_std"
            self.date_train_name = "date"
            
    def _standscaler_Xaux(self, data):
        return (data - self.X_aux_mean) / self.X_aux_std
    
    def _standscaler_X(self, data):
        data = data.astype(np.float32) / self.scale 
        data = np.where(data != 0, (data - self.X_mean) / self.X_std, 0)
        return data
    
    def _random_continuous_selection(self, X, length):
        max_start = len(X) - length
        start_index = np.random.randint(0, max_start + 1)
        return np.arange(start_index, start_index+length).astype(np.int16)
    
    def __getitem__(self, idx):
        X = self.X[idx] # seqlen, featurenum,
        seq_len, feature_num = X.shape
        missing_mask = self.missing_mask[idx] # 0 is missing, 1 is valid
        X_hat = np.copy(X)
        X_hat[missing_mask==0] = 0 # Mask original missing values
        attention_mask = np.zeros_like(X) # 1 is missing
        X_aux = self.X_aux[idx] if self.with_X_aux else np.asarray([0.]) # SAR
         
        if self.split == "train":
            # Dynamically missing 25% of the effective length of data;
            # as long as one data point can be missing, it is missing.
            if np.sum(missing_mask[:, 0]) > (1 // self.artificial_missing_rate):
                idx_orig_valid = np.where(np.all(missing_mask==1, axis=1))[0].tolist() # 1 is valid value
                idx_art_miss = np.random.choice(
                    idx_orig_valid,
                    round(len(idx_orig_valid) * self.artificial_missing_rate),
                    replace=False
                )
                X_hat[idx_art_miss, :] = 0 # mask values selected by indices and fill 0
            indicating_mask = (np.all(missing_mask==1, axis=1) ^ np.all(X_hat!=0, axis=1)).astype(np.float32) # 1 participate in impute loss cal;
            indicating_mask = np.repeat(indicating_mask[:, np.newaxis], feature_num, axis=1)
            attention_mask = np.all(X_hat==0, axis=1).astype(np.float32) # missing is 0 whether original or artifical
            
        elif self.split == "test":
            indicating_mask = self.indicating_mask[idx]
            attention_mask = np.all(X_hat==0, axis=1).astype(np.float32) 
        
        if "anytime" in self.args.mode:
            # By default, the anytime mode outputs all DOY values | 默认anytime模式训练时输出所有的DOY
            self.date_output = doy_of_year(self.args.year)
                
        # T, X, missing_mask, X_holdout, indicating_mask...
        sample = (
            torch.from_numpy(self.date_train.astype("float32")), # date for input
            torch.from_numpy(X_hat.astype("float32")), # X for input
            torch.from_numpy(missing_mask.astype("float32")), # for rec loss calculation, 0 is missing, 1 is valid observations
            torch.from_numpy(X.astype("float32")), # X_holdout for metric calculation
            torch.from_numpy(indicating_mask.astype("float32")), # for imputation loss calculation, 0 is ignore, 1 is participate
            torch.from_numpy(attention_mask.astype("float32")),  # for attention or output, 0 is valid, 1 is missing observations
            torch.from_numpy(X_aux.astype("float32")), # SAR for input
            torch.from_numpy(self.date_aux.astype("float32")), # SAR date for input
            torch.from_numpy(self.date_output.astype("float32")) # date for output (anytime)
        )
        return sample
        

class LossCollector():
    def __init__(self, ):
        self._initial()
    
    def _initial(self):
        self.imp_loss_collector = []
        self.rec_loss_collector = []
        self.tv_loss_collector = []
        self.mape_loss_collector = []
    
    def update(
        self,
        imputation_loss,
        reconstruction_loss,
        tv_loss=0,
        mape_loss=0
        ):
        self.imp_loss_collector.append(imputation_loss)
        self.rec_loss_collector.append(reconstruction_loss)
        self.tv_loss_collector.append(tv_loss)
        self.mape_loss_collector.append(mape_loss)
    
        
def train_val(model, train_dataloader, val_dataloader, optimizer, scheduler, args):
    # grad_norm_loss = GradNormLoss(optimizer, model.reduce_dim, args=args)
    best_MAE = np.inf
    for epoch in range(args.epochs):
        train_loss_collector = LossCollector()
        model.train()            
        for idx, data in enumerate(train_dataloader):
            date_input, X, missing_mask, X_holdout, indicating_mask, attention_mask, \
            X_aux, dates_aux, date_output = map(
                lambda x: x.to(args.device), data
            )
            inputs = {
                "date_input": date_input,
                "X": X,
                "missing_mask": missing_mask,
                "X_holdout": X_holdout,
                "indicating_mask": indicating_mask,
                "attention_mask": attention_mask,
                "X_aux": X_aux,
                "dates_aux": dates_aux,
                "date_output": date_output
            }
            results = model(inputs, stage="train")
            loss = results["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"] 
            
            train_loss_collector.update(
                results["imputation_loss"].data.item(),
                results["reconstruction_loss"].data.item(),
                results["total_varitation_loss"].data.item(),
            )
            
            if (idx % int(len(train_dataloader)/args.log_steps) == 0) or (idx == len(train_dataloader) - 1):
                logger.info(
                    'Epoch {:03}, {:6.2f}%, reconstruction_loss {:.5f}, imputation_loss {:.5f}, total_varitation_loss {:.5f}, lr {:.8f}'.\
                        format(
                            epoch + 1, (idx + 1)*100. / len(train_dataloader),
                            np.asarray(train_loss_collector.rec_loss_collector).mean(),
                            np.asarray(train_loss_collector.imp_loss_collector).mean(),
                            np.asarray(train_loss_collector.tv_loss_collector).mean(),
                            lr
                        )
                )
        scheduler.step()
   
        # start val      
        val_loss_collector = LossCollector()
        model.eval()
        loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        with torch.no_grad():
            for _, data in loop:
                date_input, X, missing_mask, X_holdout, indicating_mask, attention_mask, \
                X_aux, dates_aux, date_output = map(
                    lambda x: x.to(args.device), data
                )
                inputs = {
                    "date_input": date_input,
                    "X": X,
                    "missing_mask": missing_mask,
                    "X_holdout": X_holdout,
                    "indicating_mask": indicating_mask,
                    "attention_mask": attention_mask,
                    "X_aux": X_aux,
                    "dates_aux": dates_aux,
                    "date_output": date_output
                }
                results = model(inputs, stage="test")
                
                val_loss_collector.update(
                    results["imputation_loss"].data.item(),
                    results["reconstruction_loss"].data.item(),
                    mape_loss=results["mape_loss"].data.item(),
                )
                             
        info_dict = {
            "reconstruction_MAE": np.asarray(val_loss_collector.rec_loss_collector).mean(),
            "imputation_MAE": np.asarray(val_loss_collector.imp_loss_collector).mean(),
            "mean_abs_percent_error": np.asarray(val_loss_collector.mape_loss_collector).mean(),
        }      
                     
        logger.info("Epoch {}: Overall performance metrics are listed as follows:".format(epoch + 1))
        for k, v in info_dict.items():
            logger.info(f"{k}: {v}")

        if best_MAE > info_dict["imputation_MAE"]:
            best_MAE = info_dict["imputation_MAE"]
            # save model
            saving_path = os.path.join(
                args.sub_model_saving,
                "model_epoch_{}_metric_{:.5f}.ckpt".format(epoch + 1, best_MAE),
            )
            torch.save(model.state_dict(), saving_path)
            
            # # Copy the current best model to "best_model.ckpt"
            best_model_path = os.path.join(args.sub_model_saving, "best_model.ckpt")
            shutil.copyfile(saving_path, best_model_path)

    logger.info("All done. Training finished.")


def inference(model, val_dataloader, args):
    _, h, w = imread(args.ref_tif_path).shape
    args.imputed_dataset_hdf = os.path.join(args.imputed_dataset_path, "imputed_dataset.hdf")
    
    def convert_to_int16(x, scale=args.scale):
        """
        the values of original images down from GEE are between 0-10000,
        here we convert back to save memory
        """
        x = np.round(x * scale)
        return x.astype(np.int16)

    def convert_tensor_to_numpy(x):
        return x.cpu().numpy()
    
    def get_date_output(date, args=args):
        if "anytime" in args.mode and len(args.anytime_ouput) == 3:
            start_doy, end_doy, interval = args.anytime_ouput
            date_output = np.arange(start_doy, end_doy + 1, interval) 
            date_output = torch.from_numpy(date_output.astype("float32")).to(args.device)
            date_output = date_output.unsqueeze(0).repeat(date.size(0), 1)
            return date_output
        else:
            return date
    
    def write_hdf5(imputed_data, reconstructed_data, dates, args=args):
        with h5py.File(args.imputed_dataset_hdf, "a") as hf:
            imputed_group = hf.create_group("imputed_data") if "imputed_data" not in hf \
                else hf.require_group("imputed_data")
            rec_group = hf.create_group("reconstructed_data") if "reconstructed_data" not in hf \
                else hf.require_group("reconstructed_data")    
            dataset_name = str(args.ratio) if args.gap_mode == "random" else "continuous"
            if dataset_name in imputed_group:
                del imputed_group[dataset_name]
            if dataset_name in rec_group:
                del rec_group[dataset_name]
            imputed_group.create_dataset(dataset_name, data=imputed_data.astype(np.int16), compression='gzip', compression_opts=9) 
            rec_group.create_dataset(dataset_name, data=reconstructed_data.astype(np.int16), compression='gzip', compression_opts=9)
            imputed_group.create_dataset("date", data=dates.astype(np.int16), compression='gzip', compression_opts=9) 
            rec_group.create_dataset("date", data=dates.astype(np.int16), compression='gzip', compression_opts=9) 
    
    def write_tif(imputed_collector, rec_collector, date_output, args=args):
        imputed_data = rearrange(imputed_collector, "(h w) t b -> t b h w", h=h, w=w)
        rec_data = rearrange(rec_collector, "(h w) t b -> t b h w", h=h, w=w)
       
        for i in tqdm(range(len(date_output))):
            imputed_data_temp = imputed_data[i, ...]
            rec_data_temp = rec_data[i, ...]
            ymd = "".join(doy_to_ymd(year=args.year, doy=date_output[i]).split("-"))
            
            imputed_data_save_folder = os.path.join(args.imputed_dataset_path, str(args.ratio), "imputed_images")
            check_path(imputed_data_save_folder)
            imputed_data_save_path = os.path.join(imputed_data_save_folder, "S2_L2A_" + str(ymd) + ".tif")
            rec_data_save_folder = os.path.join(args.imputed_dataset_path, str(args.ratio), "reconstructed_images")
            check_path(rec_data_save_folder)
            rec_data_save_path = os.path.join(rec_data_save_folder, "S2_L2A_" + str(ymd) + ".tif")
            
            imsave(imputed_data_temp, imputed_data_save_path, dtype="int16", ref_img_path=args.ref_tif_path)
            imsave(rec_data_temp, rec_data_save_path, dtype="int16", ref_img_path=args.ref_tif_path)
                  
    imputed_collector, rec_collector, raw_collector, indicating_collector \
        = [], [], [], []
    model.eval()
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    with torch.no_grad():
        for _, data in loop:
            date_input, X, missing_mask, X_holdout, indicating_mask, attention_mask, \
            X_aux, dates_aux, date_output = map(
                lambda x: x.to(args.device), data
            )
            date_output = get_date_output(date_output)    
            inputs = {
                "date_input": date_input,
                "X": X,
                "missing_mask": missing_mask,
                "X_holdout": X_holdout,
                "indicating_mask": indicating_mask,
                "attention_mask": attention_mask,
                "X_aux": X_aux,
                "dates_aux": dates_aux,
                "date_output": date_output
            }
            stage = "anytime" if "anytime" in args.mode else "test"
            results = model(inputs, stage=stage)
            
            raw_data = convert_tensor_to_numpy(results["X_holdout"])
            imputed_data = convert_tensor_to_numpy(results["imputed_data"])
            reconstructed_data = convert_tensor_to_numpy(results["reconstructed_data"])
            date_output = convert_tensor_to_numpy(date_output).astype(np.int16)[0]
            
            raw_collector.append(raw_data)
            indicating_collector.append(convert_tensor_to_numpy(indicating_mask))
            imputed_collector.append(imputed_data)
            rec_collector.append(reconstructed_data)
            
    raw_collector = np.concatenate(raw_collector, axis=0) 
    indicating_collector = np.concatenate(indicating_collector, axis=0)
    if stage == "anytime":
        del raw_collector, indicating_collector
        imputed_collector = [convert_to_int16(img) for img in imputed_collector]
        rec_collector = [convert_to_int16(img) for img in rec_collector]
    imputed_collector = np.concatenate(imputed_collector, axis=0)
    rec_collector = np.concatenate(rec_collector, axis=0)
    
    # cal MAE, R2, RMSE ...
    if stage != "anytime" and args.cal_performance_metric:
        metric_computer = MetricNumpy(args)
        logger.info("Start calculating performance metrics...")
        metric_dict = {
            "MAE": metric_computer.masked_mae_cal(imputed_collector, raw_collector, indicating_collector),
            "RMSE": metric_computer.masked_rmse_cal(imputed_collector, raw_collector, indicating_collector),
            "R2": metric_computer.masked_r2_score(imputed_collector, raw_collector, indicating_collector),
            "MAPE": metric_computer.masked_mape(imputed_collector, raw_collector, indicating_collector),
        }
        logger.info("Performance metrics are listed as follows:")
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")    
    
    # conver to int16 to save memory
    if stage != "anytime":
        del raw_collector, indicating_collector
        imputed_collector = convert_to_int16(imputed_collector)
        rec_collector = convert_to_int16(rec_collector)
    else:
        start_doy, end_doy, interval = date_output[0], date_output[-1], date_output[1] - date_output[0]
        print(f"The reconstructed time series begins at DOY {start_doy} and ends at DOY {end_doy}, with an interval of {interval}")
    
    # logger.info("Start writing hdf dataset {}".format(args.imputed_dataset_hdf))
    # write_hdf5(imputed_collector, rec_collector, date_output)
    logger.info("Start writing tif files in folder of {}".format(args.imputed_dataset_path + os.sep + str(args.ratio)))
    write_tif(imputed_collector, rec_collector, date_output)
    logger.info("All done. Inference finished")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        default= "work_dir/Germany/base", 
        help="model and log saving dir"
    ) # 运行的结果（模型文件，日志文件，重建影像等）保存路径
    parser.add_argument(
        "--config_path",
        type=str,
        default=r"configs/Germany/base/AnytimeFormer-Germany-40%-r8-128.yaml",
        # default=r"configs/Germany/anytime/AnytimeFormer-Germany-40%-r8-128-wTV.yaml",
        help="model parameter config"
    ) # 模型参数和训练的配置文件路径
    parser.add_argument(
        "--original_dataset_path",
        type=str,
        default="dataset/dataset_for_model/Germany/hdf/random_missing.hdf",
        # default="dataset/dataset_for_model/Germany/hdf/anytime.hdf",
        help="path of original hdf dataset"
    ) # 训练和待重建数据集路径
    parser.add_argument(
        "--cal_performance_metric",
        type=str2bool,
        default=True,
        help="cal MAE, R2, RMSE for artificial masked pixels"
    ) # 是否在训练完成后接着计算精度指标（消耗大量运行内存，若不足可以设置为False）
    parser.add_argument(
        "--debug_mode",
        type=str2bool,
        default=False,
        help="Set True to use only a small dataset for quick testing" 
    ) # 是否启用debug模式，启用后只使用少量数据进行训练和测试，方便调试代码
    parser.add_argument(
        "--mode",
        type=str,
        default="train_test",
        help="mode for running", 
        choices=["train", "test", "train_test", "train_anytime", "train_test_anytime", "test_anytime"]
    ) # 运行模式选择，带有test的模式会进行测试/推理，带有anytime的模式会进行任daily重建  
    parser.add_argument(
        "--anytime_ouput",
        type=list,
        default=[1, 100, 5],
        help="Control the timestampes of outputs.", 
    ) # 控制输出的时间点，空列表表示输出一年中逐日的重建结果; 若想输出特定时间点，可指定[1, 101, 5]，1表示起始DOY，101表示终止DOY，5表示时间间隔 
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="outputs loss every how many iterations" 
    ) # 训练过程中每多少个iteration输出一次loss  
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1024,
        help="inference_batchsize during inference"
    ) # 测试/推理时的batch size
    parser.add_argument(
        "--scale",
        type=int,
        default=10000,
        help="convert images down from GEE (u/int16) to float"
    ) # 将影像数据从u/int16转换为float时的缩放系数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run model" 
    ) # 模型运行设备选择，cuda或cpu
    parser.add_argument(
        "--ref_tif_path",
        type=str,
        default="dataset/dataset_for_model/Germany/tif/40%/S2_L2A_20190208.tif",
        help="Used to obtain geographic coordinate information when writing out imputed images."
    ) # 输出重建影像时的参考tif影像路径，用于获取地理坐标系统，仿射变换参数等，选择原始缺失数据集中的一张影像即可
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="",
        help="for continuousGap to improve performance"
    ) # 预训练模型路径，可选择使用预训练模型进行微调以提升性能
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="work_dir/Germany/best_model_anytime.ckpt",
        help="test mode to test saved model"
    ) # 测试/推理时加载的模型路径
    parser.add_argument(
        "--gap_doy",
        type=str,
        default="",
        help="for continuous missing experiments, e.g., 100_105"
    ) # 连续缺失模式下指定缺失的DOY范围，例如100_105表示缺失第100到105天的数据
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)
    sys.path.append(os.getcwd())

    # parameters
    seed_torch()
    args = parser.parse_args()
    with open(args.config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "test" in args.mode:
        assert (
            os.path.exists(args.ref_tif_path) 
        ), "ref_tif_path must be existed"        
        if args.mode == "test":
            assert (
                args.saved_model_path is not None 
            ), "saved_model_path must be provided in test mode"
            
    # merge parameters
    # priority paraser reading
    for section in cfg:
        cfg_args = cfg[section]
        for key, value in cfg_args.items():
            if not hasattr(args, key):
                setattr(args, key, value)

    # create dirs
    time_now = datetime.now().__format__("%Y-%m-%d_T%H:%M:%S").replace(':', '-')
    exp_name = os.path.basename(args.config_path).split(".")[0]
    log_saving = os.path.join(args.work_dir, time_now + "_" + exp_name)
    args.sub_model_saving = os.path.join(args.work_dir,  time_now + "_" + exp_name, "models")
    dir_list = [args.sub_model_saving, log_saving]
    if "test" in args.mode:
        args.imputed_dataset_path = os.path.join(args.work_dir, time_now + \
        "_" + str(exp_name), "inference")
        dir_list.append(args.imputed_dataset_path)
        if "train" not in args.mode:
            dir_list.remove(args.sub_model_saving)

    [
        os.makedirs(dir_)
        for dir_ in dir_list
        if not os.path.exists(dir_)
    ]
    
    # create logger
    logger = setup_logger(os.path.join(log_saving, "log.log"), "w")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
         
    # build models and dataloaders 
    args.log_steps = 1 if args.debug_mode else args.log_steps
    if "train" in args.mode:
        train_dataset = MyDataset(
            args.original_dataset_path,
            "train",
            args=args
        )
        train_dataloader = DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else 0,
        )
    val_dataset = MyDataset(
        args.original_dataset_path,
        "test",
        args=args,
    )
    val_dataloader = DataLoader(
        val_dataset,
        args.inference_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else 0,
    )
    # File naming convention for output TIF files
    if args.ref_tif_path is not None:
        args.year = int(os.path.basename(args.ref_tif_path).split("_")[-1][0:4])
    else:
        args.year = None
    
    # for reversed standard of model's outputs
    args.X_mean = torch.from_numpy(val_dataset.X_mean).to(args.device)
    args.X_std = torch.from_numpy(val_dataset.X_std).to(args.device)
    
    model = model_dict[args.model_name](args)
    model = model.to(args.device)
    
    # training
    if "train" in args.mode:
        logger.info("Start training...")
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.epochs // 5 if args.epochs >= 5 else 1,
            gamma=0.9
        )
        if args.debug_mode:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        if os.path.exists(args.pretrained_model_path):
            load_model(model, args.pretrained_model_path, logger)    
        train_val(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
        if "test" in args.mode:
            saved_model_path = os.path.join(args.sub_model_saving, "best_model.ckpt")
            logger.info("Start loading model from best_checkpoint {}".format(saved_model_path))
            load_model(model, saved_model_path, logger)
            logger.info("Start testing...")
            inference(model, val_dataloader, args)
            
    # testing/inference
    elif  "test" in args.mode:
        logger.info("Start loading model...")
        load_model(model, args.saved_model_path, logger)
        logger.info("Start inferencing...")
        inference(model, val_dataloader, args)