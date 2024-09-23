from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from visualize_attention import visualize_image_attention
model_path = "/home/luoyiwen/scratch/LLaVA/llava-v1.5-7b"
# model_path = "/home/luoyiwen/scratch/LLaVA/llava-v1.6-34b"
prompt = "Please describe this image in detail."
image_file = "/home/luoyiwen/scratch/LLaVA/images/dish_som.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
# visualize_image_attention(args)
