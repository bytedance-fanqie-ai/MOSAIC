import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed
import numpy as np
from torchvision import transforms
from datasets import load_dataset

from my_datasets.Subject200k_dataset import Subjects200K, make_collate_fn


def get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=None, img_path=None):
    
    if img_path is not None:
        feature_base = img_path.replace('JPEGImages', 'features').replace('.jpg', '')
        sd_path = f"{feature_base}_sd.pt"
        dino_path = f"{feature_base}_dino.pt"

    # extract stable diffusion features
    if img_path is not None and os.path.exists(sd_path):
        features_sd = torch.load(sd_path)
        for k in features_sd:
            features_sd[k] = features_sd[k].to('cuda')
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_sd_input = resize(img, target_res=num_patches*16, resize=True, to_pil=True)
        features_sd = process_features_and_mask(sd_model, sd_aug, img_sd_input, mask=False, raw=True)
        del features_sd['s2']

    # extract dinov2 features
    if img_path is not None and os.path.exists(dino_path):
        features_dino = torch.load(dino_path)
    else:
        if img is None: img = Image.open(img_path).convert('RGB')
        img_dino_input = resize(img, target_res=num_patches*14, resize=True, to_pil=True)
        img_batch = extractor_vit.preprocess_pil(img_dino_input)
        features_dino = extractor_vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, num_patches, num_patches)

    desc_gathered = torch.cat([
            features_sd['s3'],
            F.interpolate(features_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(features_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            features_dino
        ], dim=1)
    
    desc = aggre_net(desc_gathered) # 1, 768, 60, 60
    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


set_seed(42)
num_patches = 64
sd_model = sd_aug = extractor_vit = None
aggre_net = AggregationNetwork(feature_dims=[640,1280,1280,768], projection_dim=768, device='cuda')
aggre_net.load_pretrained_weights(torch.load('results_spair/best_856.PTH'))

sd_model, sd_aug = load_model(diffusion_ver='v1-5', image_size=num_patches*16, num_timesteps=50, block_indices=[2,5,8,11])
extractor_vit = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')


img_size = num_patches * 16
latent_size = num_patches



to_pil = transforms.ToPILImage()



def process_dataset(dataset):
    # === 确定背景目录 ===
    coords_dir = "/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/coord"

    coords_sub_dir = os.path.join(coords_dir, "geoaware")
    os.makedirs(coords_sub_dir, exist_ok=True)
    # === DataLoader ===
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=make_collate_fn(num_refs=6),  # 你的 collate_fn
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 这里关掉 DataLoader 多进程
        pin_memory=True,  # 不固定到 GPU
        drop_last=False,
    )

    for batch in tqdm(train_dataloader, desc=f"Processing {dataset.__class__.__name__}"):
        ids = batch['ids']
        # Move tensors to GPU and normalize to [0, 1]
        ref_imgs = ((batch['ref_imgs'] + 1) / 2.).cuda()
        tgt_img = ((batch['tgt_imgs'][0] + 1) / 2.).cuda()
        masks = batch['masks'][0].cuda()

        id = ids[0]
        instance_num = ref_imgs.shape[1]

        for i in range(instance_num):
            if os.path.exists(f"{coords_sub_dir}/{id}_ref{i}.pt"):
                print(f"{coords_sub_dir}/{id}_ref{i}.pt already exists")
                continue
            
            ref_img_tensor = ref_imgs[0, i] # from reference images
            mask_tensor = masks[i]          # from masks

            if mask_tensor.sum() == 0:
                continue

            # Image for feature 1: target image with character mask
            feat1_img_tensor = tgt_img * mask_tensor
            feat1_img_pil = to_pil(feat1_img_tensor.cpu())
            feat1_img_pil = feat1_img_pil.resize((img_size, img_size))

            # Image for feature 2: reference image with white background removed
            ref_img_permuted = ref_img_tensor.permute(1, 2, 0)
            white = torch.tensor([1.0, 1.0, 1.0], device=ref_img_permuted.device, dtype=ref_img_permuted.dtype)
            tolerance = 10 / 255.0
            is_white = torch.all(torch.abs(ref_img_permuted - white) <= tolerance, dim=-1)
            bg_mask = ~is_white
            feat2_img_tensor = ref_img_permuted * bg_mask.unsqueeze(-1)
            feat2_img_tensor = feat2_img_tensor.permute(2, 0, 1)
            feat2_img_pil = to_pil(feat2_img_tensor.cpu())
            feat2_img_pil = feat2_img_pil.resize((img_size, img_size))

            # Get features
            feat1 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=feat1_img_pil)
            feat2 = get_processed_features(sd_model, sd_aug, aggre_net, extractor_vit, num_patches, img=feat2_img_pil)

            B, C, H, W = feat1.shape

            # 1. 展平空间维度 -> [H*W, C]
            feat1_flat = feat1.view(C, -1).T  # [H*W, C]
            feat2_flat = feat2.view(C, -1).T  # [H*W, C]

            # 3. 计算相似度矩阵 -> [H*W, H*W]
            cos_map = torch.mm(feat1_flat, feat2_flat.T)

            # 4. 对每个位置找到最相似的点
            max_sim, max_idx = torch.max(cos_map, dim=1)  # [H*W]

            # 5. 将 idx 转换成 (y, x) 坐标
            max_y = (max_idx // W).unsqueeze(1) 
            max_x = (max_idx % W).unsqueeze(1) 

            # 6. 拼成 (H*W, 2) 的矩阵
            coords = torch.cat([max_y, max_x], dim=1)  # [H*W, 2]

            torch.save(coords, f"{coords_sub_dir}/{id}_ref{i}.pt")



if __name__ == "__main__":

    # Dataset and DataLoaders creation ===============================
    data_files = {"train":os.listdir("/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/data")}
    dataset_subject200k = load_dataset("parquet", data_dir="/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/data", data_files=data_files)
    def filter_func(item):
        if item.get("collection") != "collection_2":
            return False
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    dataset_subject200k_valid = dataset_subject200k["train"].filter( # 过滤出高质量
        filter_func,
        num_proc=32,
        # cache_file_name="./cache/dataset/collection_2_valid.arrow", # Optional
    )
    # 使用自定义类
    subject_dataset = Subjects200K(
        original_dataset=dataset_subject200k_valid,
        ref_size=1024,
        tgt_size=1024,
        grounding_dir="/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/mask",
        mode="train", t_drop_rate=0.00, i_drop_rate=0.00, ti_drop_rate=0.00, # dropout for cfg
        add_postfix=False,
    )
    
    dataset_list = [
        subject_dataset, # 11w
    ]

    for ds in dataset_list:
        process_dataset(ds)