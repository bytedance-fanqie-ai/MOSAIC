import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import PILToTensor
from PIL import Image, ImageDraw
from tqdm import tqdm
from datasets import load_dataset


from src.models.dift_sd import SDFeaturizer
from my_datasets.Subject200k_dataset import Subjects200K, make_collate_fn


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def resize(img, target_res=224, resize_img=True, to_pil=True, edge=False):
    """
    Resizes and pads an image to a square canvas.
    """
    original_width, original_height = img.size
    
    if resize_img:
        aspect_ratio = original_width / original_height
        if original_width >= original_height:
            new_width = target_res
            new_height = int(np.around(target_res / aspect_ratio))
        else:
            new_height = target_res
            new_width = int(np.around(target_res * aspect_ratio))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    width, height = img.size
    img_np = np.asarray(img)
    
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    canvas_shape = [target_res, target_res, channels] if channels > 1 else [target_res, target_res]
    canvas = np.zeros(canvas_shape, dtype=np.uint8)

    if not edge:
        pad_top = (target_res - height) // 2
        pad_left = (target_res - width) // 2
        canvas[pad_top:pad_top + height, pad_left:pad_left + width] = img_np
    else:
        pad_top = (target_res - height) // 2
        pad_bottom = target_res - height - pad_top
        pad_left = (target_res - width) // 2
        pad_right = target_res - width - pad_left
        pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if channels > 1:
            pad_width.append((0,0))
        canvas = np.pad(img_np, pad_width, mode='edge')

    return Image.fromarray(canvas) if to_pil else canvas

def remove_white_background(img_tensor, tolerance=5):
    """
    Removes white background from an image tensor.
    """
    img_np = np.array(img_tensor) * 255
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    white = np.array([255, 255, 255])
    is_white = np.all(np.abs(img_np - white) <= tolerance, axis=-1)
    mask = ~is_white
    
    img_masked = img_np * mask[..., None] / 255.0
    return to_pil(img_masked)

def extract_features(dift, image, ensemble_size):
    """
    Extracts features from an image using the DIFT model.
    """
    img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
    return dift.forward(img_tensor, prompt="", ensemble_size=ensemble_size)

def calculate_dift_coordinates(feat1, feat2):
    """
    Calculates matching coordinates between two feature maps.
    """
    C, H, W = feat1.shape
    feat1_flat = feat1.view(C, -1).T
    feat2_flat = feat2.view(C, -1).T
    
    cos_map = torch.mm(feat1_flat, feat2_flat.T)
    max_sim, max_idx = torch.max(cos_map, dim=1)
    
    max_y = (max_idx // W).unsqueeze(1)
    max_x = (max_idx % W).unsqueeze(1)
    
    return torch.cat([max_y, max_x], dim=1)

def visualize_matches(coords_dift, coords_geo_path, valid_idx, latent_size, ref_img, tgt_img, img_size, save_path_prefix):
    """
    Visualizes the matches between two sets of coordinates and saves a single combined image.
    """
    if not os.path.exists(coords_geo_path):
        print(f"Geo-aware coordinates file not found: {coords_geo_path}")
        return

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    coords_geo = torch.load(coords_geo_path)

    H, W = latent_size, latent_size
    match = (coords_dift[valid_idx] == coords_geo[valid_idx]).all(dim=1)

    ratio = match.float().mean().item()
    print(f"Match ratio: {ratio:.4f}")

    valid_idx_cpu = valid_idx.cpu()
    match_cpu = match.cpu()
    match_idx = valid_idx_cpu.nonzero(as_tuple=False).squeeze(1)[match_cpu]

    ref_y = (match_idx // W).tolist()
    ref_x = (match_idx % W).tolist()
    tgt_y = coords_dift[match_idx, 0].cpu().tolist()
    tgt_x = coords_dift[match_idx, 1].cpu().tolist()
    matches_list = list(zip(ref_y, ref_x, tgt_y, tgt_x))

    scale = img_size // latent_size
    ref_img_full = resize(to_pil(tgt_img).convert("RGB"), target_res=img_size, resize_img=True, to_pil=True)
    tgt_img_full = resize(to_pil(ref_img).convert("RGB"), target_res=img_size, resize_img=True, to_pil=True)

    draw_ref = ImageDraw.Draw(ref_img_full)
    draw_tgt = ImageDraw.Draw(tgt_img_full)

    for idx, (ry, rx, ty, tx) in enumerate(matches_list):
        ry_img, rx_img = ry * scale + scale // 2, rx * scale + scale // 2
        ty_img, tx_img = ty * scale + scale // 2, tx * scale + scale // 2

        draw_ref.ellipse((rx_img-3, ry_img-3, rx_img+3, ry_img+3), outline="red", width=3)
        draw_tgt.ellipse((tx_img-3, ty_img-3, tx_img+3, ty_img+3), outline="blue", width=3)
        draw_ref.text((rx_img+5, ry_img-5), str(idx), fill="red")
        draw_tgt.text((tx_img+5, ty_img-5), str(idx), fill="blue")

    # 拼接两张图：横向拼接
    combined = Image.new("RGB", (ref_img_full.width + tgt_img_full.width, img_size))
    combined.paste(ref_img_full, (0, 0))
    combined.paste(tgt_img_full, (ref_img_full.width, 0))

    combined.save(f"{save_path_prefix}_matches.png")
    print(f"Saved combined visualization to {save_path_prefix}_matches.png")


def process_dataset(dataset, dift, args):
    # === 确定背景目录 ===
    coords_dir = "/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/coord"

    parent_name = os.path.basename(os.path.dirname(coords_dir))
    coords_sub_dir = os.path.join(coords_dir, "dift")
    os.makedirs(coords_sub_dir, exist_ok=True)
    # === DataLoader ===
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=make_collate_fn(num_refs=6),  # 你的 collate_fn
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 这里关掉 DataLoader 多进程
        pin_memory=True,  # 不固定到 GPU
        drop_last=False,
    )

    for batch in tqdm(dataloader, desc=f"Processing {dataset.__class__.__name__}"):
        ids = batch['ids']
        ref_imgs = ((batch['ref_imgs'] + 1) / 2.).cpu()
        tgt_img = ((batch['tgt_imgs'][0] + 1) / 2.).cpu()
        masks = batch['masks'][0].cpu()
        
        instance_num = ref_imgs.shape[1]
        id = ids[0]
        for i in range(instance_num):
            save_path = f"{coords_sub_dir}/{id}_ref{i}.pt"
            if os.path.exists(save_path):
                print(f"Skipping, already exists: {save_path}")
                continue
                
            ref_img = ref_imgs[0][i]
            mask = masks[i]
            if mask.sum() == 0:
                continue
            
            print(f"Processing: {id}, instance {i}")
            ref_img_masked = to_pil(tgt_img * mask).resize((args.img_size, args.img_size))
            tgt_img_masked = remove_white_background(ref_img).resize((args.img_size, args.img_size))

            # 只在对应区域内点匹配，提高准确率
            feat1 = extract_features(dift, ref_img_masked, args.ensemble_size).squeeze(0)
            feat2 = extract_features(dift, tgt_img_masked, args.ensemble_size).squeeze(0)
            
            coords = calculate_dift_coordinates(feat1, feat2)
            torch.save(coords, save_path)
            print(f"Saved coordinates to: {save_path}")

            latent_size = args.img_size // 16
            # 如果是 uint8 且 max==1，则先放大到 0/255
            if mask.dtype == torch.uint8 and mask.max() <= 1:
                mask = mask * 255
            ref_mask_pil = to_pil(mask).resize((latent_size, latent_size)).convert("L")
            ref_mask_np = np.array(ref_mask_pil, dtype=np.uint8)
            ref_mask_bin = (ref_mask_np > 127).astype(np.uint8)

            valid_idx = torch.from_numpy(ref_mask_bin.flatten()).bool()
            coords_geo_path = coords_sub_dir.replace("dift", "geoaware") + f"/{id}_ref{i}.pt"

            try:
                visualize_matches(
                    coords_dift=coords,
                    coords_geo_path=coords_geo_path,
                    valid_idx=valid_idx,
                    latent_size=latent_size,
                    ref_img=ref_img,
                    tgt_img=tgt_img,
                    img_size=args.img_size,
                    save_path_prefix=f"./visualization/{parent_name}/{id}_{i}"
                )
            except Exception as e:
                print(f"Error visualize_matches {id}_{i}: {e}")


def main():
    parser = argparse.ArgumentParser(description="DIFT Feature Matching and Visualization")
    parser.add_argument('--sd_model_path', type=str, default="/mnt/bn/shedong/hf_model/stabilityai/stable-diffusion-2-1", help="Path to Stable Diffusion model.")
    parser.add_argument('--img_size', type=int, default=1024, help="Image size for processing.")
    parser.add_argument('--ensemble_size', type=int, default=8, help="Ensemble size for DIFT.")
    parser.add_argument('--device', type=int, default=0, help="CUDA device to use.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    
    global to_pil
    to_pil = transforms.ToPILImage()

    dift = SDFeaturizer(sd_id=args.sd_model_path)

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
        process_dataset(ds, dift, args)


if __name__ == "__main__":
    main()