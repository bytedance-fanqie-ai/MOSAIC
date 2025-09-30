import os
import random
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import hashlib
from io import BytesIO
import traceback
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_image_hash(image: Image.Image) -> str:
    """Convert a PIL image to a SHA256 hash string."""
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
    return hashlib.sha256(img_bytes).hexdigest()[:8]  # 取前8位更短但足够唯一


def build_ref2tgt_mapping(
        dift_coord, geoaware_coord, mask, 
        tgt_patch_size=64, ref_patch_size=32
    ):
    """
    Args:
        dift_coord: Tensor [N, 2], tgt_patch -> ref_patch (y, x) on ref grid
        geoaware_coord: Tensor [N, 2], tgt_patch -> ref_patch (y, x) on ref grid
        mask: Tensor [1, H, W] (binary), tgt mask (already resized to tgt_size)
        tgt_patch_size: int, tgt patch grid resolution (default 64 for 1024/16)
        ref_patch_size: int, ref patch grid resolution (if None, assume same as tgt)

    Returns:
        ref2tgt: dict {ref_patch_id: [tgt_patch_id, ...]}
    """
    N = dift_coord.shape[0]   # tgt patch count
    assert dift_coord.shape == geoaware_coord.shape

    if ref_patch_size is None:
        ref_patch_size = tgt_patch_size  # 默认和 tgt 相同

    # --- 1. 找出交集位置 ---
    valid_mask = torch.all(dift_coord == geoaware_coord, dim=1)  # [N] bool

    # --- 2. resize mask 到 tgt patch 空间 ---
    mask_patch = F.interpolate(mask.unsqueeze(0), size=(tgt_patch_size, tgt_patch_size),
                            mode="nearest")[0,0]   # [tgt_patch_size, tgt_patch_size]
    mask_flat = mask_patch.view(-1) > 0   # [N] bool

    # --- 3. tgt patch 中有效的索引 = 两个条件同时成立
    keep_mask = valid_mask & mask_flat.to(device=dift_coord.device)
    keep_idx = torch.arange(N, device=dift_coord.device)[keep_mask]  # [K], 对应 tgt 中对应物体有效 idx 区域

    # 4. ref 坐标 (64-grid → target_ref_patch_size-grid)
    ref_coords64 = dift_coord[keep_idx]  # [K,2]
    ref_coords_new = (ref_coords64 * ref_patch_size) // 64
    ref_patch_idx = ref_coords_new[:,0] * ref_patch_size + ref_coords_new[:,1]

    # 5. 构建反映射
    ref2tgt = {}
    for r, t in zip(ref_patch_idx.tolist(), keep_idx.tolist()):
        if r not in ref2tgt:   # 只保留第一个
            ref2tgt[r] = t

    return ref2tgt


class Subjects200K(Dataset):
    def __init__(
        self, 
        original_dataset, 
        mode, 
        ref_size, 
        tgt_size, 
        grounding_dir=None,
        coord_folder: str=None,
        padding=8, 
        img_size=512, 
        t_drop_rate=0.05, 
        i_drop_rate=0.05, 
        ti_drop_rate=0.05,
        add_postfix=False
    ):
        self.original_dataset = original_dataset
        self.grounding_dir = grounding_dir
        self.coord_folder = coord_folder    
        self.mode = mode
        self.padding = padding
        self.ref_size = ref_size # resize 大小
        self.tgt_size = tgt_size # resize 大小
        self.img_size = img_size # 原图大小
        self.caption_prefix = [
                        "The two-panel image presents a xx in various scenes.", 
                        "The two-panel image showcases a xx in different scenes.",
                        "The two-panel image emphasizes a xx in different scenarios.",
                        "The two-panel image visuals illustrate a xx in diverse scenes.",
                        "The two-panel image features a xx in different scenarios."
                    ]
        
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.add_postfix = add_postfix
        self.transform_ref = transforms.Compose([   
            transforms.Resize(self.ref_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.ToTensor(), # [0, 255] --> [0, 1]
            transforms.Normalize([0.5], [0.5]), # [0, 1] --> [-1, 1]
        ])

        self.transform_tgt = transforms.Compose([   
            transforms.Resize(self.tgt_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias = True),
            transforms.ToTensor(), # [0, 255] --> [0, 1]
            transforms.Normalize([0.5], [0.5]), # [0, 1] --> [-1, 1]
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.tgt_size, interpolation=transforms.InterpolationMode.NEAREST),  # 保留 mask 边界
            transforms.ToTensor(),  # 转为 [0,1] 的 float tensor, shape: [1, H, W]
            transforms.Lambda(lambda x: (x > 0.5).to(torch.uint8))  # 二值化，0 或 1
        ])

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        try:
            example = self.original_dataset[idx]
            image = example['image']
            description = example['description']
            item = description['item']
            tgt_caption = description['description_0']
            ref_caption = description['description_1']
            
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")

            tgt_img_pil = image.crop(
                (self.padding, self.padding, self.img_size + self.padding, self.img_size + self.padding)
            )
            ref_img_pil = image.crop(
                (
                    self.img_size + self.padding * 2,
                    self.padding,
                    self.img_size * 2 + self.padding * 2,
                    self.img_size + self.padding,
                )
            )
            # ref_img = self.transform(ref_img)
            ref_img = self.transform_ref(ref_img_pil)
            tgt_img = self.transform_tgt(tgt_img_pil)

            # 加入 image hash 到 key 中
            try:
                img_hash = get_image_hash(tgt_img_pil)
            except Exception as e:
                raise ValueError(f"Failed to hash image: {e}")
            key = f"{item}_{tgt_caption}_{img_hash}"
            uid = hashlib.sha256(key.encode("utf-8")).hexdigest()
            mask_path = os.path.join(self.grounding_dir, f"{uid}.png")
            if os.path.exists(mask_path):
                mask_pil = Image.open(mask_path)
                mask = self.transform_mask(mask_pil)
            else:
                mask = torch.ones((1, self.tgt_size, self.tgt_size), dtype=torch.uint8)
                # raise ValueError(f"Mask not found for item: {item}, caption: {tgt_caption}")

            coords = []
            dift_coord_path = os.path.join(self.coord_folder, "dift", f"{uid}_ref0.pt")
            geoaware_coord_path = os.path.join(self.coord_folder, "geoaware", f"{uid}_ref0.pt")
            if os.path.exists(dift_coord_path) and os.path.exists(geoaware_coord_path):
                dift_coord = torch.load(dift_coord_path, map_location="cpu", weights_only=True)
                geoaware_coord = torch.load(geoaware_coord_path, map_location="cpu", weights_only=True)

                # Resize coordinates
                orig_h, orig_w = 64, 64
                new_h, new_w = self.tgt_size // 16, self.tgt_size // 16

                if (new_h, new_w) != (orig_h, orig_w):
                    # dift_coord
                    dift_coord = dift_coord.reshape(1, orig_h, orig_w, 2).permute(0, 3, 1, 2).float()
                    dift_coord = F.interpolate(dift_coord, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    dift_coord = dift_coord.permute(0, 2, 3, 1).reshape(new_h * new_w, 2).long()
                    
                    # geoaware_coord
                    geoaware_coord = geoaware_coord.reshape(1, orig_h, orig_w, 2).permute(0, 3, 1, 2).float()
                    geoaware_coord = F.interpolate(geoaware_coord, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    geoaware_coord = geoaware_coord.permute(0, 2, 3, 1).reshape(new_h * new_w, 2).long()


                ref2tgt_coords = build_ref2tgt_mapping(dift_coord, geoaware_coord, mask, tgt_patch_size=self.tgt_size // 16, ref_patch_size=self.ref_size // 16)
                coords.append(ref2tgt_coords)
            else:
                # print(f"Coord not found for uid: {uid}")
                coords.append({})
            
            caption = tgt_caption
            drop_image = False
            drop_text = False
            drop_mask = False
            rand_num = random.random()
            if "train" in self.mode:
                if rand_num < self.i_drop_rate: # 0~0.05 drop image
                    drop_image = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate): # 0.05~0.10 drop text
                    drop_text = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate): # 0.10~0.15 drop image text mask
                    drop_image = True
                    drop_text = True
                    drop_mask = True
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate + self.i_drop_rate): # 0.15~0.20 drop mask
                    drop_mask = True

            result = {
                'id': uid,
                'ref_imgs': [ref_img],
                'tgt_img': tgt_img,
                'bboxs': [[0, 0, self.tgt_size, self.tgt_size]],
                'masks': [mask],
                'coords': coords,
                'caption': caption,
                'drop_image': drop_image,
                'drop_text': drop_text,
                'drop_mask': drop_mask
            }

            return result

        except Exception as e:
            print(f"加载索引 {idx} 失败: {e}")
            traceback.print_exc()
            # 随机选择一个新的索引以避免无限递归
            new_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(new_index)


def make_collate_fn(num_refs: int = 3):
    """
    返回一个带参数的 collate_fn，以便 DataLoader 中简单写:
        collate_fn = make_collate_fn(num_refs=3)
    """
    def _collate(examples):
        # ---- 1. 先确定图像基础形状 (C,H,W) ----
        # 以第一条样本的 tgt_img 作为参考
        c, h, w = examples[0]["ref_imgs"][0].shape
        pad_ref_img = torch.zeros(c, h, w, dtype=examples[0]["ref_imgs"][0].dtype)

        # ---- 2. 收集并 pad / truncate 参考图 ----
        ids = []
        ref_imgs = []
        tgt_imgs = []
        captions = []
        masks = []
        drop_images = []
        drop_texts = []
        drop_masks = []

        for ex in examples:
            refs = ex["ref_imgs"]                     # list[Tensor] 长度 0~num_refs
            refs = refs[:num_refs]                  # 若多于 num_refs 则截断

            ref_imgs.append(torch.stack(refs))      # (num_refs, C, H, W)

            ids.append(ex["id"])
            tgt_imgs.append(ex["tgt_img"])
            captions.append(ex["caption"])
            # --- 处理 masks ---
            ex_masks = ex["masks"][:num_refs]
            if len(ex_masks) < num_refs:
                # 获取目标图大小 (1, C, H, W) -> (H, W)
                _, H, W = ex["tgt_img"].shape
                pad_mask = torch.zeros((1, H, W), dtype=torch.uint8)
                ex_masks += [pad_mask] * (num_refs - len(ex_masks))
            masks.append(torch.stack(ex_masks))  # (num_refs, H, W)

            drop_images.append(ex["drop_image"])
            drop_texts.append(ex["drop_text"])
            drop_masks.append(ex["drop_mask"])

        # ---- 3. 堆 batch 维 ----
        ref_imgs = torch.stack(ref_imgs)            # (B, num_refs, C, H, W)
        tgt_imgs = torch.stack(tgt_imgs)            # (B, C, H, W)
        masks = torch.stack(masks)

        batch = {
            "ids"        : ids,
            "ref_imgs"   : ref_imgs,                # 统一 tensor
            "tgt_imgs"   : tgt_imgs,
            "captions"   : captions,                # 仍保持 list[str]
            "masks"      : masks,
            "drop_images": torch.as_tensor(drop_images, dtype=torch.bool),
            "drop_texts" : torch.as_tensor(drop_texts, dtype=torch.bool),
            "drop_masks" : torch.as_tensor(drop_masks, dtype=torch.bool),
        }
        return batch

    return _collate


def make_collate_fn_w_coord(num_refs: int = 3):
    """
    返回一个带参数的 collate_fn，以便 DataLoader 中简单写:
        collate_fn = make_collate_fn(num_refs=3)
    """
    def _collate(examples):
        # ---- 1. 先确定图像基础形状 (C,H,W) ----
        # 以第一条样本的 tgt_img 作为参考
        c, h, w = examples[0]["ref_imgs"][0].shape
        pad_ref_img = torch.zeros(c, h, w, dtype=examples[0]["ref_imgs"][0].dtype)

        # ---- 2. 收集并 pad / truncate 参考图 ----
        ids = []
        ref_imgs = []
        tgt_imgs = []
        captions = []
        masks = []
        coords = []
        drop_images = []
        drop_texts = []
        drop_masks = []

        for ex in examples:
            refs = ex["ref_imgs"]                     # list[Tensor] 长度 0~num_refs
            refs = refs[:num_refs]                  # 若多于 num_refs 则截断
            ref_imgs.append(torch.stack(refs))      # (num_refs, C, H, W)
            
            ids.append(ex["id"])
            tgt_imgs.append(ex["tgt_img"])
            captions.append(ex["caption"])
            coords.append(ex["coords"])
            # --- 处理 masks ---
            ex_masks = ex["masks"][:num_refs]
            if len(ex_masks) < num_refs:
                # 获取目标图大小 (1, C, H, W) -> (H, W)
                _, H, W = ex["tgt_img"].shape
                pad_mask = torch.zeros((1, H, W), dtype=torch.uint8)
                ex_masks += [pad_mask] * (num_refs - len(ex_masks))
            masks.append(torch.stack(ex_masks))  # (num_refs, H, W)

            drop_images.append(ex["drop_image"])
            drop_texts.append(ex["drop_text"])
            drop_masks.append(ex["drop_mask"])

        # ---- 3. 堆 batch 维 ----
        ref_imgs = torch.stack(ref_imgs)            # (B, num_refs, C, H, W)
        tgt_imgs = torch.stack(tgt_imgs)            # (B, C, H, W)
        masks = torch.stack(masks)

        batch = {
            "ids"        : ids,
            "ref_imgs"   : ref_imgs,                # 统一 tensor
            "tgt_imgs"   : tgt_imgs,
            "captions"   : captions,                # 仍保持 list[str]
            "masks"      : masks,
            "coords"     : coords,
            "drop_images": torch.as_tensor(drop_images, dtype=torch.bool),
            "drop_texts" : torch.as_tensor(drop_texts, dtype=torch.bool),
            "drop_masks" : torch.as_tensor(drop_masks, dtype=torch.bool),
        }
        return batch

    return _collate



if __name__ == "__main__":

    import os
    import random
    from datasets import load_dataset

    data_files = {"train":os.listdir("/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/data")}
    dataset = load_dataset(
        "parquet", 
        data_dir="/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/data", 
        data_files=data_files,
        # features=features
    )["train"]
    def filter_func(item):
        if item.get("collection") != "collection_2":
            return False
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    dataset_valid = dataset.filter(filter_func, num_proc=16)

    ref_size = 1024
    tgt_size = 1024
    # 使用自定义类
    custom_train = Subjects200K(
        original_dataset=dataset_valid,
        mode="train",
        ref_size=ref_size,
        tgt_size=tgt_size,
        grounding_dir="/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/mask",
        coord_folder="/mnt/bn/shedong/hf_data/Yuanshi/Subjects200K/coord",
    )

    train_dataloader = torch.utils.data.DataLoader(
        custom_train, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True, 
        collate_fn=make_collate_fn(num_refs=1),
    )

    from tqdm.auto import tqdm
    from torchvision.utils import save_image
    to_pil_image = transforms.ToPILImage()

    def patch_id_to_xy(patch_id, grid_size, image_size):
        """把 patch_id 转换成图像坐标 (x,y)"""
        y = patch_id // grid_size
        x = patch_id % grid_size
        stride = image_size / grid_size
        return int((x + 0.5) * stride), int((y + 0.5) * stride)

    def visualize_mapping_with_lines(idx, ref_img, tgt_img, ref2tgt,
                                    ref_image_size=512, tgt_image_size=1024,
                                    ref_grid_size=32, tgt_grid_size=64,
                                    max_pairs=100):
        """
        可视化 ref->tgt 的 patch 映射：在拼接图上画点并连线
        """
        ref_img = np.array(ref_img)
        tgt_img = np.array(tgt_img)

        # 把两张图拼到一起 (水平拼接)
        h = max(ref_img.shape[0], tgt_img.shape[0])
        w = ref_img.shape[1] + tgt_img.shape[1]
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        canvas[:ref_img.shape[0], :ref_img.shape[1]] = ref_img
        canvas[:tgt_img.shape[0], ref_img.shape[1]:] = tgt_img

        pairs = []
        for r, t in ref2tgt.items():
            pairs.append((r, t))
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

        for (r, t) in pairs:
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # ref 点坐标
            rx, ry = patch_id_to_xy(r, ref_grid_size, ref_image_size)
            # tgt 点坐标（要加上 ref 图的宽度，才能映射到拼接图）
            tx, ty = patch_id_to_xy(t, tgt_grid_size, tgt_image_size)
            tx += ref_img.shape[1]

            # 画点
            cv2.circle(canvas, (rx, ry), 4, color, -1)
            cv2.circle(canvas, (tx, ty), 4, color, -1)

            # 连线
            cv2.line(canvas, (rx, ry), (tx, ty), color, 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(canvas)
        plt.axis("off")
        plt.title("Ref ↔ Tgt Patch Mapping")
        plt.savefig(f"ref2tgt_mapping_ref{idx}.png")

    # 示例
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training Batches", total=len(train_dataloader))):
        ids = batch['ids']
        print("id: ", ids)
        caption = batch['captions']
        ref = (batch['ref_imgs'] + 1) / 2.
        tgt = (batch['tgt_imgs'] + 1) / 2.
        mask = batch['masks'].squeeze(2)
        coords = batch['coords'][0]
        from torchvision.utils import save_image
        for i in range(len(coords)):
            ref_img = ref[:, i]
            save_image(ref_img, f'ref_img{i}.jpg')
            
            visualize_mapping_with_lines(
                    i, to_pil_image(ref_img.squeeze(0)), to_pil_image(tgt.squeeze(0)), coords[i],
                    ref_image_size=ref_size, tgt_image_size=tgt_size,
                    ref_grid_size=ref_size // 16, tgt_grid_size=tgt_size // 16
                )

        save_image(tgt, 'tgt.jpg')
        save_image(mask.float(), 'mask.jpg')
        print(caption)