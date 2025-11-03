import os
import sys
sys.path.append(os.getcwd())
import numpy as np

from glob import glob
from tqdm import tqdm
from osgeo import gdal
from utils import imsave, imread


def remove_cloud_from_s2_images(
    site: str,
    base_folder: str = "./dataset/dataset_down_from_GEE",
    valid_percent: float = 0.2,
    use_scl: int = 1,
    use_cp: int = 0,
    use_cs: int = 0,
    thre_cp: int = 50,
    thre_cs: int = 50,
    
):
    """
    Remove cloud pixels from Sentinel-2 images based on cloud masks (SCL / Cloud Probability / Cloud Score).
    基于云掩膜（SCL / Cloud Probability / Cloud Score）去除 Sentinel-2 影像中的云像元。

    Args:
        site (str): Site name, e.g., "Germany". 处理站点名称，例如 "Germany"。
        base_folder (str): Root dataset folder path. 数据集根目录路径。
        valid_percent (float): Minimum valid (non-cloud) pixel ratio threshold, default 0.2 (20%). 最低非云像元比例阈值（默认 0.2 即 20%）。
        use_scl (int): Whether to use SCL mask (1=use / 0=not use). 是否使用 SCL 掩膜（1 使用 / 0 不使用）。
        use_cp (int): Whether to use Cloud Probability mask. 是否使用 Cloud Probability 掩膜。
        use_cs (int): Whether to use Cloud Score mask. 是否使用 Cloud Score 掩膜。
        thre_cp (int): Cloud probability threshold (values > threshold are considered cloud, default 50). 云概率阈值，大于该值视为云（默认 50）。
        thre_cs (int): Cloud score threshold (values > threshold are considered cloud, default 50). 清晰像素概率阈值，小于该值视为云（默认 50）。
    """

    # ==== Input and output path configuration ====
    # ==== 输入与输出路径配置 ====
    s2_input_folder = os.path.join(base_folder, site, "S2_raw")
    output_folder = os.path.join(base_folder, site, "S2_remove_cloud_1")
    os.makedirs(output_folder, exist_ok=True)

    s2_cp_file_prefix = os.path.join(base_folder, site, "Cloud_Probability", "S2_Cloud_Probability")
    s2_cs_file_prefix = os.path.join(base_folder, site, "Cloud_Score", "S2_Cloud_Score")

    s2_filelist = glob(os.path.join(s2_input_folder, "*.tif"))
    if not s2_filelist:
        print(f"[Error] No image files found in {s2_input_folder}. 未在路径 {s2_input_folder} 下找到影像文件。")
        return

    print(f"Start processing site: {site} / 开始处理站点：{site}")
    print(f"Number of images: {len(s2_filelist)} / 影像数量：{len(s2_filelist)}，输出路径：{output_folder}")

    for s2_file in tqdm(s2_filelist, desc="Processing S2 images / 处理 Sentinel-2 影像"):
        date = os.path.basename(s2_file).split("_")[-1].split(".")[0]
        s2_image = imread(s2_file)[0:-1, :, :]  # Remove last SCL band / 除去最后一层 SCL 波段

        # ========== 1. Read and process SCL cloud mask ==========
        # ========== 1. 读取并处理 SCL 云掩膜 ==========
        s2_scl = imread(s2_file)[-1, :, :]
        conditions_scl = (s2_scl == 1) | (s2_scl == 2) | (s2_scl == 3) | \
                         (s2_scl == 7) | (s2_scl == 8) | (s2_scl == 9) | (s2_scl == 10)
        s2_scl_mask = np.where(conditions_scl, 1, 0)

        # ========== 2. Cloud Probability mask ==========
        # ========== 2. Cloud Probability 掩膜 ==========
        s2_cp_file = f"{s2_cp_file_prefix}_{date}.tif"
        if os.path.exists(s2_cp_file):
            s2_cp = imread(s2_cp_file)
            s2_cp_mask = np.where(s2_cp > thre_cp, 1, 0)
        else:
            s2_cp_mask = np.zeros_like(s2_scl_mask)

        # ========== 3. Cloud Score mask ==========
        # ========== 3. Cloud Score 掩膜 ==========
        s2_cs_file = f"{s2_cs_file_prefix}_{date}.tif"
        if os.path.exists(s2_cs_file):
            s2_cs = imread(s2_cs_file)
            s2_cs1, s2_cs2 = s2_cs[0, :, :], s2_cs[1, :, :]
            s2_cs1[np.isnan(s2_cs1)] = 0
            s2_cs2[np.isnan(s2_cs2)] = 0
            s2_cs_mask = np.where((s2_cs1 < thre_cs) | (s2_cs2 < thre_cs), 1, 0)
        else:
            s2_cs_mask = np.zeros_like(s2_scl_mask)

        # ========== 4. Combine all cloud masks ==========
        # ========== 4. 合并所有云掩膜 ==========
        s2_cloud_mask = s2_scl_mask * use_scl + s2_cp_mask * use_cp + s2_cs_mask * use_cs
        cloud_pixels = (s2_cloud_mask > 0)

        # ========== 5. Determine whether to keep this image ==========
        # ========== 5. 判断是否保留该影像 ==========
        non_cloud_ratio = 1 - (np.sum(cloud_pixels) / (s2_image.shape[1] * s2_image.shape[2]))
        if non_cloud_ratio < valid_percent:
            print(f"Skip {os.path.basename(s2_file)} (non-cloud ratio={non_cloud_ratio:.2f}) / 跳过 {os.path.basename(s2_file)}（非云像元比例={non_cloud_ratio:.2f}）")
            continue

        # ========== 6. Remove cloud pixels and save ==========
        # ========== 6. 去除云像元并保存 ==========
        s2_image_no_cloud = np.where(cloud_pixels, 0, s2_image)
        output_path = os.path.join(output_folder, os.path.basename(s2_file))
        imsave(s2_image_no_cloud, output_path, dtype="uint16", ref_img_path=s2_file)

    print("Cloud removal finished! 云去除完成！")


def main():
    print("Starting cloud removal process... / 开始执行云去除流程...")
    remove_cloud_from_s2_images(
        site="Germany",
        valid_percent=0.2,
        use_scl=1,
        use_cp=0,
        use_cs=0,
        thre_cp=50,
        thre_cs=50,
    )
    print("All processes completed! 所有处理已完成！")


if __name__ == "__main__":
    main()
