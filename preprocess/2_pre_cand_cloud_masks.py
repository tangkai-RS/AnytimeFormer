import os
import sys
import numpy as np

from tqdm import tqdm
from glob import glob
from typing import List

sys.path.append(os.getcwd())
from utils import imsave, imread


def generate_candidate_cloud_masks(
    site: str,
    base_folder: str = "./dataset/dataset_down_from_GEE",
    cand_mask_folders: List[str] = None,
    valid_percent: float = 0.2,
    max_cloud_percent: float = 0.9,
    threshold: int = 50
):
    """
    Generate standardized binary cloud masks based on candidate masks (e.g., Cloud Probability).
    基于候选云掩膜（如 Cloud Probability）生成标准化的二值云掩膜。

    Args:
        site (str): Site name, e.g., "Germany". 站点名称（例如 "Germany"）。
        base_folder (str): Root dataset directory path. 数据集根目录路径。
        cand_mask_folders (List[str]): List of candidate cloud mask folders. 候选云掩膜文件夹列表。
        valid_percent (float): Minimum acceptable cloud coverage ratio (default 0.2). 最小云像元比例阈值（默认 0.2）。
        max_cloud_percent (float): Maximum acceptable cloud coverage ratio (default 0.9). 最大云像元比例阈值（默认 0.9）。
        threshold (int): Cloud probability threshold (values > threshold are considered cloud, default 50). 云概率阈值，大于该值视为云（默认 50）。
    """

    if cand_mask_folders is None:
        cand_mask_folders = ["Cloud_Probability"]

    # ==== Output folder path ====
    # ==== 输出文件夹路径 ====
    save_folder = os.path.join(base_folder, site, "cand_cloud_mask", "cloud_mask")
    os.makedirs(save_folder, exist_ok=True)

    # ==== Collect all candidate mask files ====
    # ==== 收集所有候选掩膜文件 ====
    s2_cp_filelist = []
    for folder_name in cand_mask_folders:
        folder_path = os.path.join(base_folder, site, "cand_cloud_mask", folder_name)
        s2_cp_filelist.extend(glob(os.path.join(folder_path, "*.tif")))

    if not s2_cp_filelist:
        print(f"[Error] No candidate cloud mask files found in {cand_mask_folders}. 未在 {cand_mask_folders} 中找到候选云掩膜文件。")
        return

    print(f"Site: {site} / 站点：{site}")
    print(f"Number of candidate cloud mask source files: {len(s2_cp_filelist)} / 候选云掩膜源文件数：{len(s2_cp_filelist)}")
    print(f"Output directory: {save_folder} / 输出目录：{save_folder}")
    print("Start generating candidate cloud masks... / 开始生成候选云掩膜...\n")

    # ==== Process each cloud mask file ====
    # ==== 处理每一个云掩膜文件 ====
    for s2_cp_file in tqdm(s2_cp_filelist, desc="Generating cloud masks / 生成云掩膜"):
        s2_cp = imread(s2_cp_file)

        # Step 1: Binarize mask (values > threshold are cloud)
        # Step 1: 二值化掩膜（大于阈值的视为云）
        s2_cp_mask = np.where(s2_cp > threshold, 1, 0).astype(np.uint8)

        # Step 2: Calculate cloud coverage ratio
        # Step 2: 计算云覆盖比例
        cloud_percent = np.sum(s2_cp_mask) / (s2_cp_mask.shape[0] * s2_cp_mask.shape[1])

        # Step 3: Determine if within acceptable range
        # Step 3: 判断是否在可接受范围内
        if valid_percent < cloud_percent < max_cloud_percent:
            output_path = os.path.join(save_folder, os.path.basename(s2_cp_file))
            imsave(s2_cp_mask, output_path, dtype="uint8", ref_img_path=s2_cp_file)
        else:
            print(f"Skip {os.path.basename(s2_cp_file)} (cloud ratio={cloud_percent:.2f}) / 跳过 {os.path.basename(s2_cp_file)}（云覆盖率={cloud_percent:.2f}）")

    print("\nCandidate cloud mask generation completed! 候选云掩膜生成完成！")


def main():
    print("Starting candidate cloud mask generation... / 开始候选云掩膜生成流程...")
    generate_candidate_cloud_masks(
        site="Germany",
        cand_mask_folders=[
            # for Germany
            "S2-cloud_pro-Gre_2020",
            "S2-cloud_pro-Gre_2021",
            "S2-cloud_pro-Gre_2022",
            "S2-cloud_pro-Gre_2024",
            "Cloud_Probability"
        ],
        valid_percent=0.2,
        max_cloud_percent=0.9,
        threshold=50
    )
    print("All processes completed! 所有处理已完成！")


if __name__ == "__main__":
    main()
