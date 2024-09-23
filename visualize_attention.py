import pickle

import argparse
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def visualize_attention(attentions: torch.Tensor, image_file: str, model_name: str):
    if attentions.dim() == 2:
        attentions = attentions.unsqueeze(dim=1)
    # (patch(anyres), num_heads, spatial_token_num)
    assert attentions.dim() == 3, "Dim of attention is not supported!"
    # print(f"attentions shape for single image: {attentions.shape}")
    num_heads = attentions.shape[1]
    pixel_size = 336
    patch_size = 14
    output_dir = "./attentions/{}/{}".format(model_name, os.path.splitext(os.path.basename(image_file))[0])
    os.makedirs(output_dir, exist_ok=True)
    for i in range(attentions.shape[0]):
        attention = attentions[i].reshape(num_heads, -1)
        w_heatmap, h_heatmap = int(pixel_size / patch_size), int(pixel_size / patch_size)
        attention = attention.reshape(num_heads, h_heatmap, w_heatmap)
        attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        # save attentions heatmaps
        for j in range(num_heads):
            fname = os.path.join(output_dir, "patch_" + str(i) + "_attn-head_" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attention[j], format='png')
            print(f"{fname} saved.")

        attention = torch.sum(attention, dim=0)
        fname = os.path.join(output_dir, "patch_" + str(i) + "_attn-sum.png")
        plt.imsave(fname=fname, arr=attention, format='png')
        print(f"{fname} saved.")


def visualize_features(features: torch.Tensor, image_file: str, model_name: str):
    if features.dim() == 2:
        features = features.unsqueeze(dim=0)
    # (patch(anyres), spatial_token_num, hidden_dim)
    assert features.dim() == 3, "Dim of features is not supported!"
    # print(f"attentions shape for single image: {attentions.shape}")
    pixel_size = 336
    patch_size = 14
    w_heatmap, h_heatmap = int(pixel_size / patch_size), int(pixel_size / patch_size)
    spatial_token_num = w_heatmap * h_heatmap
    output_dir = "./features/{}/{}".format(model_name, os.path.splitext(os.path.basename(image_file))[0])
    os.makedirs(output_dir, exist_ok=True)
    for i in range(features.shape[0]):
        feature = features[i].reshape(spatial_token_num, -1).norm(dim=-1)
        feature = feature.reshape((h_heatmap, w_heatmap))
        feature = nn.functional.interpolate(feature.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0, 0].cpu()
        select_layer = 24 + (-2) + 1
        fname = os.path.join(output_dir, "patch_" + str(i) + "_layer_" + str(select_layer) + "_feature.png")
        plt.imsave(fname=fname, arr=feature, format='png')
        print(f"{fname} saved.")


def visualize_image_attention(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    image_files = image_parser(args)
    images = load_images(image_files)  # list of RGB images
    image_sizes = [x.size for x in images]  # list of (width, height)
    print(f"image_sizes: {image_sizes}")
    images = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    # save preprocessed image
    if type(images) is list:
        # TODO: save multiple images
        print("")
    else:
        assert images.dim() == 4
        output_dir = "./preprocessed_images/"
        os.makedirs(output_dir, exist_ok=True)
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        mean = np.array(image_mean)
        std = np.array(image_std)
        for i in range(images.shape[0]):
            img_array = images[i].permute(1, 2, 0).cpu().numpy()
            img_array = img_array * std + mean
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
            image.save(os.path.join(output_dir, os.path.splitext(os.path.basename(image_files[i]))[0] + ".png"))
        print("preprocessed images are saved!")

    with torch.inference_mode():
        vision_tower = model.get_vision_tower()
        output_attentions = True
        if images is None:
            raise ValueError(f"No image is input")
        elif type(images) is list or images.ndim == 5:  # llava-1.6-34b <anyres>
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            # 将image_num和image_patch维度合并为batch维度, [(5,3,336,336),(3,3,336,336)]->(8,3,336,336)
            concat_images = torch.cat([image for image in images], dim=0)
            # (8,3,336,336) -> (8,576,dim)
            image_outputs = vision_tower(concat_images, output_attentions=output_attentions)
            image_features, image_attentions = image_outputs["image_features"], image_outputs["image_attentions"]
            # [5,3]
            split_sizes = [image.shape[0] for image in images]
            # (8, 576, dim) -> ([5, 576, dim], [3, 576, dim])
            # image_features = torch.split(image_features, split_sizes, dim=0)
            # (8, 16, 576, 576) -> [(5, 16, 576, 576), [3, 16, 576, 576)]
            image_attentions = torch.split(image_attentions, split_sizes, dim=0)
        else:  # llava-1.5-7b <pad>
            image_outputs = vision_tower(images, output_attentions=output_attentions)
            image_features, image_attentions = image_outputs["image_features"], image_outputs["image_attentions"]
        if isinstance(image_attentions, tuple):
            # TODO: image_attentions包含多张输入图片
            print("to implement...")
        else:
            visualize_attention(image_attentions, image_files[0], model_name)
            print("visualization of image attentions success!")
            visualize_features(image_features, image_files[0], model_name)
            print("visualization of image features success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    visualize_image_attention(args)
