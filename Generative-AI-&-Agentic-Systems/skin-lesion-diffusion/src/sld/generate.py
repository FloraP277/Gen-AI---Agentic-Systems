"""Generate synthetic images for one rare class with a LoRA-fine-tuned Stable Diffusion.

    python -m sld.generate --lora work/lora/df --dx df --n 300 --out work/synthetic

Each image gets its own seed, so a run can be reproduced exactly.
"""
import argparse
import os

from .data import prompt_for

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True, help="Folder with the trained LoRA weights")
    p.add_argument("--dx", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default="work/synthetic")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args(argv)

    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16).to("cuda")
    pipe.load_lora_weights(args.lora)
    pipe.set_progress_bar_config(disable=True)
    out = os.path.join(args.out, args.dx)
    os.makedirs(out, exist_ok=True)
    prompt = prompt_for(args.dx)
    for start in range(0, args.n, args.batch):
        seeds = list(range(start, min(start + args.batch, args.n)))
        gens = [torch.Generator("cuda").manual_seed(s) for s in seeds]
        images = pipe([prompt] * len(seeds), num_inference_steps=args.steps, guidance_scale=args.guidance,
                      generator=gens, height=512, width=512).images
        for s, img in zip(seeds, images):
            img.save(os.path.join(out, f"synthetic_{args.dx}_{s:05d}.jpg"), quality=95)
    print(f"wrote {args.n} images to {out}")


if __name__ == "__main__":
    main()
