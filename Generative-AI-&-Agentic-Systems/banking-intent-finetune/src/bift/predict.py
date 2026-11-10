"""Batched greedy generation with a fine-tuned adapter, with timing."""
import time

import pandas as pd

from .data import prompt_messages
from .labels import parse_prediction


def load_model(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or base)
    tok.padding_side = "left"   # needed for batched generation with decoder-only models
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def predict(df: pd.DataFrame, tok, model, intents: list, batch: int = 32) -> pd.DataFrame:
    import torch

    texts = df["text"].tolist()
    outputs, start = [], time.perf_counter()
    for i in range(0, len(texts), batch):
        prompts = [tok.apply_chat_template(prompt_messages(t), tokenize=False, add_generation_prompt=True)
                   for t in texts[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    seconds = (time.perf_counter() - start) / max(len(texts), 1)
    return df.assign(raw=outputs,
                     pred_intent=[parse_prediction(o, intents) for o in outputs],
                     seconds_per_message=seconds)
