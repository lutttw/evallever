# -*- coding: utf-8 -*-
"""修订66-B:Qwen2.5-7B NF4 温度采样推理(云端,与 p15 cmd_infer 逐字段同构)。"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--suffix', type=str, required=True)
a = ap.parse_args()

MODEL = '<cloud-root>/qwen2.5-7b-instruct'
PROMPTS = '<cloud-root>/prompts_en_json.jsonl'
OUTF = f'<cloud-root>/preds_qwen2.5-7b-instruct_en_json{a.suffix}.jsonl'
BATCH, MAX_NEW, MAX_PROMPT = 8, 768, 7000

torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                       bnb_4bit_compute_dtype=torch.float16,
                       bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q,
                                             device_map='cuda:0',
                                             trust_remote_code=True)
model.eval()

rows = [json.loads(l) for l in open(PROMPTS, encoding='utf-8') if l.strip()]
print(f'[samp{a.suffix}] {len(rows)} prompts', flush=True)

t0, n_done = time.time(), 0
with open(OUTF, 'w', encoding='utf-8') as fo:
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        texts = []
        for r in batch:
            msgs = [{'role': 'user', 'content': r['prompt']}]
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(t, add_special_tokens=False)['input_ids']
            if len(ids) > MAX_PROMPT:
                t = tok.decode(ids[:MAX_PROMPT])
                t = t.rsplit('[S', 1)[0] + '\n(输出截断。)'
            texts.append(t)
        enc = tok(texts, return_tensors='pt', padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=0.7, top_p=1.0, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
        for r, inp, o in zip(batch, enc['input_ids'], out):
            fo.write(json.dumps({'did': r['did'], 'title': r['title'],
                                 'model': 'qwen2.5-7b-instruct', 'fmt': 'json',
                                 'load': '4bit', 'ents': r['ents'],
                                 'raw': tok.decode(o[len(inp):], skip_special_tokens=True)},
                                ensure_ascii=False, sort_keys=True) + '\n')
            fo.flush()
        n_done += len(batch)
        dt = (time.time() - t0) / n_done
        print(f'[samp{a.suffix}] {n_done}/{len(rows)} {dt:.1f}s/doc eta {(len(rows)-n_done)*dt/3600:.1f}h',
              flush=True)
print(f'[samp{a.suffix}] done', flush=True)
