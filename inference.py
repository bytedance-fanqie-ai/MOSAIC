import os
import json
import torch
from torchvision import transforms
from PIL import Image
from diffusers import FluxPipeline

from src.flux_omini import Condition, generate
from utils import process_image


device = "cuda"
dtype = torch.bfloat16

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
).to(device)


to_tensor = transforms.Compose([
            transforms.ToTensor(), # [0, 255] --> [0, 1]
        ])


pipe.load_lora_weights(
    "ByteDance-FanQie/MOSAIC",
    weight_name=f"subject_512.safetensors",
    adapter_name="subject"
)
pipe.set_adapters(["subject"], [1])


max_num_refs = 6
ref_size = 512
height = 512
width = 512

guidance_scale=3.5


# 创建输出目录（如果不存在）
out_dir = './outputs'
os.makedirs(out_dir, exist_ok=True)


with open(f'example_cases.json', 'r', encoding='utf-8') as f:
    data_list = json.load(f)

    for item in data_list:
        index = item['index']
        prompt = item['prompt']
        print("prompt:", prompt)
        image_paths = item['image_paths']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        ref_imgs = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                pil_img = process_image(img_path, target_size=ref_size, pad_color=(255,255,255), scale=0.9)
            else:
                pil_img = Image.new("RGB", (ref_size, ref_size), (0,0,0))
                print(f"{img_path} not exists, all black")
            ref_imgs.append(pil_img)


        position_deltas = []
        for i in range(len(ref_imgs)):
            position_deltas.append([0, -(ref_size * (i + 1)) // 16])
            
        conditions = [Condition(appearance, "subject", position_deltas[i]) for i, appearance in enumerate(ref_imgs)]
        
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=prompt,
                conditions=conditions,
                num_inference_steps=28,
                num_images_per_prompt=1,
                height=height,
                width=width,
                guidance_scale=guidance_scale, # if not args.de_distill else None, # 训练的时候为 1, 推理的时候为 3.5
                generator=torch.Generator("cuda").manual_seed(42),
            )[0]
        if len(result) == 0:
            print(f"警告: 生成结果为空，跳过 {index}")
            continue
            
        result_img = result[0]
        result_img_path = os.path.join(out_dir, f"{index}_cfg_{guidance_scale}_{height}x{width}.jpg")
        result_img.save(result_img_path)
        print(f"保存生成的图像到 {result_img_path}")

        # 调整参考图像大小
        resized_refs = [pil.resize((height, width), Image.Resampling.LANCZOS) for pil in ref_imgs]
        # 拼接图像
        panel = len(image_paths)
        concat_image = Image.new("RGB", (height * (panel + 1), height))
        for i, pil_img in enumerate(resized_refs[:panel]):
            concat_image.paste(pil_img, (height * i, 0))
        concat_image.paste(result_img, (height * panel, 0))
        concat_image_path = os.path.join(out_dir, f"{index}_{height}x{width}_compared.jpg")
        concat_image.save(concat_image_path)
        print(f"保存对比图像到 {concat_image_path}")