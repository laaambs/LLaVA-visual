import argparse
import os
import pickle

import numpy as np
import torch
import math

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import Levenshtein
import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(image_file, sep):
    out = image_file.split(sep)
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


def calculate_similarity(seq1, seq2):
    return 1 - Levenshtein.distance(seq1, seq2) / max(len(seq1), len(seq2))


# 计算一致性分数
def calculate_consistency_counts(candidate_sequences, consistency_threshold):
    n = len(candidate_sequences)
    consistency_counts = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            similarity = calculate_similarity(
                candidate_sequences[i], candidate_sequences[j]
            )
            if similarity > consistency_threshold:
                consistency_counts[i] += 1
                consistency_counts[j] += 1

    return torch.tensor([float(count) for count in consistency_counts])


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    total_results = []
    for query, image_file in zip(args.query, args.image_file):
        qs = query
        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        conv.system = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, accurate, and honest answers to the human's questions. The assistant respond with \"I'dont know\" when it doesn't have confident answer."

        image_files = image_parser(image_file, sep=args.sep)
        images = load_images(image_files)  # list of RGB images
        image_sizes = [x.size for x in images]  # list of (width, height)
        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                # do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                # top_p=args.top_p,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,  # 返回所有束
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                # output_attentions=True,
                output_scores=True,  # 输出分数
                return_dict_in_generate=True,  # 返回字典格式的输出
            )

        # print(f"tuple_attentions: {type(outputs.attentions)}, {len(outputs.attentions)}")
        # print(f"tuple_attentions[0]: {type(outputs.attentions[0])}, {len(outputs.attentions[0])}")
        # print(f"tuple_attentions[0][0]: {type(outputs.attentions[0][0])}, {outputs.attentions[0][0].shape}")
        # output_tokens = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

        # output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
        #     0
        # ].strip()
        # print(f"outputs: {output_tokens}")

        candidate_sequences = tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        # 添加一些调试信息
        print(f"output.sequences shape: {output.sequences.shape}")
        print(f"len(output.scores): {len(output.scores)}")
        print(f"output.scores[0] shape: {output.scores[0].shape}")

        # 计算每个候选序列的初始对数概率 (seq_len, num_beams, 32000)
        init_log_dist = output.scores[0][0, :].log_softmax(-1)
        init_indices = output.sequences[:, 1]
        init_log_probs = init_log_dist[init_indices].to(
            dtype=torch.float16, device=model.device
        )

        # 计算每个候选序列的平均对数概率
        avg_log_scores = output.sequences_scores.to(
            dtype=torch.float16, device=model.device
        )

        # 计算长度补偿
        sequence_lengths = (output.sequences[:, 1:] != tokenizer.pad_token_id).sum(
            dim=1
        )
        length_compensations = torch.log1p(sequence_lengths.float()).to(
            dtype=torch.float16, device=model.device
        )

        # # 使用min-max归一化对数概率
        # init_log_probs_norm = self.min_max_normalize(init_log_probs).to(dtype=torch.float16, device=self.model.device)
        # avg_scores_norm = self.min_max_normalize(output.sequences_scores).to(dtype=torch.float16, device=self.model.device)

        # 使用softmax归一化对数概率
        # init_probs_norm = F.softmax(init_log_probs, dim=0).to(dtype=torch.float16, device=self.model.device)
        # avg_scores_norm = F.softmax(output.sequences_scores, dim=0).to(dtype=torch.float16, device=self.model.device)

        # seq_lengths_norm = self.min_max_normalize(sequence_lengths).to(dtype=torch.float16, device=self.model.device)

        # 计算一致性分数
        # consistency_scores = self.calculate_consistency_scores(candidate_sequences, consistency_threshold).to(dtype=torch.float16, device=self.model.device)
        consistency_scores = torch.log1p(
            calculate_consistency_counts(
                candidate_sequences, args.consistency_threshold
            )
        ).to(dtype=torch.float16, device=model.device)

        token_probs = []
        for i in range(args.num_beams):
            seq_probs = []
            for j, score in enumerate(output.scores):
                if j + 1 == len(output.sequences[i]):
                    break
                # 获取当前位置的token id
                token_id = output.sequences[i, j + 1].item()
                # 计算该token的概率
                prob = score[i, token_id].exp().item()
                seq_probs.append(prob)
            token_probs.append(seq_probs)

        candidate_probs = []
        for candidate_sequence, token_prob in zip(output.sequences, token_probs):
            tokens = tokenizer.convert_ids_to_tokens(candidate_sequence[1:])
            candidate_probs.append({"sequence": tokens, "probs": token_prob})

        confident_scores = (
            args.alpha * init_log_probs
            + args.beta * avg_log_scores
            + args.gamma * length_compensations
            + args.delta * consistency_scores
        )

        results = []
        for (
            seq,
            init_score,
            avg_score,
            seq_length,
            length_compensation,
            consistency_score,
            confident_score,
        ) in zip(
            candidate_sequences,
            init_log_probs,
            avg_log_scores,
            sequence_lengths,
            length_compensations,
            consistency_scores,
            confident_scores,
        ):
            results.append(
                {
                    "sequence": seq.strip(),
                    "init_score": init_score.item(),
                    "avg_score": avg_score.item(),
                    "seq_length": seq_length.item(),
                    "length_compensation": length_compensation.item(),
                    "consistency_score": consistency_score.item(),
                    "confident_score": confident_score.item(),
                }
            )

        total_results.append(
            {
                "prompt": query,
                "scored_candidates": sorted(
                    results, key=lambda x: x["confident_score"], reverse=True
                ),
                "candidate_probs": candidate_probs,
            }
        )
    return total_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
