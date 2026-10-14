"""Load a Qwen3-VL model and run a quick image-text inference sanity check."""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--image_path", required=True)
    p.add_argument("--prompt", default="Describe the image.")
    args = p.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()

    from PIL import Image

    img = Image.open(args.image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    input_len = inputs.input_ids.shape[1]
    out = processor.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
    print("=" * 40)
    print(out)


if __name__ == "__main__":
    main()
