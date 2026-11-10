"""Supervised fine-tuning (SFT) with LoRA on Banking77. Runs on a free Colab T4 GPU.

LoRA trains small adapter matrices instead of all the model's weights, so a
0.5B-parameter model fits comfortably in 16 GB of GPU memory.

    python -m bift.train_sft --model Qwen/Qwen2.5-0.5B-Instruct --out runs/sft
"""
import argparse
import json
import os

from .data import load_banking77, split_train, to_prompt_completion


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="runs/sft")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-train", type=int, default=None, help="Limit examples for a quick trial run")
    args = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train, _, intents = load_banking77()
    parts = split_train(train)
    sft = parts["sft"] if args.max_train is None else parts["sft"].sample(args.max_train, random_state=1)
    train_ds = Dataset.from_list(to_prompt_completion(sft))
    val_ds = Dataset.from_list(to_prompt_completion(parts["val"]))

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if bf16 else torch.float16)

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20,
        eval_strategy="steps", eval_steps=100, save_strategy="epoch",
        bf16=bf16, fp16=not bf16, report_to="none", seed=42,
    )
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "adapter"))
    tok.save_pretrained(os.path.join(args.out, "adapter"))

    os.makedirs(args.out, exist_ok=True)
    json.dump({"base_model": args.model, "train_examples": len(sft), "epochs": args.epochs,
               "lr": args.lr, "lora_r": args.lora_r, "log_history": trainer.state.log_history},
              open(os.path.join(args.out, "training.json"), "w"), indent=1)
    json.dump(intents, open(os.path.join(args.out, "intents.json"), "w"))


if __name__ == "__main__":
    main()
