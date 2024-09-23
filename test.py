from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava2 import eval_model
from visualize_attention import visualize_image_attention
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_candidate_norms(results, file_name, id=id, save_dir="visualization_results",):
    candidate_norms = [result["norms"] for result in results]
    candidate_tokens = [result["token_seq"] for result in results]
    num_candidates = len(results)
    fig, axs = plt.subplots(
        num_candidates, 1, figsize=(25, 5 * num_candidates), squeeze=False
    )

    # 创建保存目录(如果不存在)
    os.makedirs(save_dir, exist_ok=True)

    for i, norms in enumerate(candidate_norms):
        token_sequence = candidate_tokens[i]

        # 确保tokens和probs长度一致
        min_len = min(len(token_sequence), len(norms))
        tokens = token_sequence[:min_len]
        norms = norms[:min_len]

        # 创建柱状图
        axs[i, 0].bar(range(len(norms)), norms)
        axs[i, 0].set_ylim(0, 150) 
        axs[i, 0].set_title(f"Candidate {i+1}")
        axs[i, 0].set_ylabel("L2-Norm")

        # 在x轴上显示token
        axs[i, 0].set_xticks(range(len(token_sequence)))
        axs[i, 0].set_xticklabels(token_sequence, rotation=45, ha="right")

    # 调整布局,防止重叠
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(
        save_dir, f"candidate_norms_{file_name}_{id}_{len(candidate_norms)}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # 关闭图形,释放内存

    print(f"Visualization saved to {save_path}")

CAPTION_PROMPT = """Give a clear and concise summary of the image above in one paragraph. In your summary, you need to describe the relative position of each person appearing in the image (either on the left, in the middle, or on the right) and explain their action as briefly as possible. """

# model_path = "liuhaotian/llava-v1.5-7b"
# model_path = "/content/drive/MyDrive/LLaVA/llava-v1.5-7b"
model_path = "/home/luoyiwen/scratch/LLaVA/llava-v1.5-7b"
# model_path = "/data/coding/models/llava-v1.5-7b"
prompt = [
    CAPTION_PROMPT,
    CAPTION_PROMPT,
    CAPTION_PROMPT,
    # "What is the image a photo of?",
    # "What is the total number of people in the room?",
    # "What are the people in the middle doing?",
    # "What objects are in the room?",
    # "What do the scattered books indicate about the people's activity?",
]
image_file = [
    "/home/luoyiwen/scratch/LLaVA/images/val-92.jpg",
    "/home/luoyiwen/scratch/LLaVA/images/val-164.jpg",
    "/home/luoyiwen/scratch/LLaVA/images/val-253.jpg",
    # "/home/luoyiwen/scratch/LLaVA/images/val-92.jpg",
    # "/home/luoyiwen/scratch/LLaVA/images/val-92.jpg",
    # "/home/luoyiwen/scratch/LLaVA/images/val-164.jpg",
    # "/home/luoyiwen/scratch/LLaVA/images/val-253.jpg",
    # "/home/luoyiwen/scratch/LLaVA/images/val-400.jpg",
]


args = type(
    "Args",
    (),
    {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 6,
        "num_beam_groups": 2,
        "diversity_penalty": 0.5,
        "max_new_tokens": 512,
        "alpha": 0.15,
        "beta": 0.8,
        "gamma": 0.02,
        "delta": 0.03,
        "consistency_threshold": 0.67,
    },
)()

total_results = eval_model(args)

for id, results in enumerate(total_results):
    print(f"prompt {id}: {results['prompt']}")
    print(results["scored_candidates"])
    # file_name = os.path.basename(image_file[id])
    # file_name = os.path.splitext(file_name)[0]
    # visualize_candidate_norms(results["scored_candidates"], file_name=file_name, id=id)
# visualize_image_attention(args)
