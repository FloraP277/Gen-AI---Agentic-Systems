"""Optional stage 2: preference tuning with DPO, starting from the SFT model.

1. Run the SFT model on the held-out `pref` slice of the training split.
2. Turn its mistakes into (prompt, chosen = correct intent, rejected = its wrong answer) pairs.
3. Merge the SFT adapter into the base model and train a new LoRA adapter with DPO.
   With ref_model=None, TRL uses the merged SFT model with the new adapter
   switched off as the reference, which is exactly the SFT model.

DPO is a simpler alternative to RLHF: it learns from preference pairs directly,
without training a separate reward model.

    python -m bift.train_dpo --sft runs/sft --out runs/dpo
"""
import argparse
import json
import os

import pandas as pd

from .data import load_banking77, split_train
from .pairs import build_pairs
from .predict import load_model, predict


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="runs/sft")
    p.add_argument("--out", default="runs/dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1)
    args = p.parse_args(argv)

    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    meta = json.load(open(os.path.join(args.sft, "training.json")))
    intents = json.load(open(os.path.join(args.sft, "intents.json")))
    train, _, _ = load_banking77()
    pref = split_train(train)["pref"]

    tok, model = load_model(meta["base_model"], os.path.join(args.sft, "adapter"))
    preds = predict(pref.rename(columns={"intent": "true_intent"}), tok, model, intents)
    pairs = build_pairs(preds)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(preds).drop(columns=["raw"]).to_csv(os.path.join(args.out, "pref_predictions.csv"), index=False)
    print(f"{len(pairs)} preference pairs from {len(pref)} held-out messages")
    if not pairs:
        print("The SFT model made no mistakes on the held-out slice; nothing to learn from.")
        return

    merged = model.merge_and_unload()
    tok.padding_side = "right"
    cfg = DPOConfig(output_dir=args.out, beta=args.beta, learning_rate=args.lr, num_train_epochs=args.epochs,
                    per_device_train_batch_size=8, logging_steps=10, save_strategy="no",
                    report_to="none", fp16=True, seed=42)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=merged, ref_model=None, args=cfg, train_dataset=Dataset.from_list(pairs),
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(os.path.join(args.out, "adapter"))
    merged.save_pretrained(os.path.join(args.out, "sft_merged"))
    tok.save_pretrained(os.path.join(args.out, "sft_merged"))
    json.dump({"base_model": os.path.join(args.out, "sft_merged"), "pairs": len(pairs),
               "beta": args.beta, "log_history": trainer.state.log_history},
              open(os.path.join(args.out, "training.json"), "w"), indent=1)
    json.dump(intents, open(os.path.join(args.out, "intents.json"), "w"))


if __name__ == "__main__":
    main()
