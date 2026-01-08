import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

phrase = "Artificial intelligence is fascinating."
inputs = tokenizer(phrase, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab)

# TODO: convertir en probabilités (softmax)
probs = torch.softmax(logits, dim=-1)

# On affiche P(token_t | tokens_) pour t>=1
input_ids = inputs["input_ids"][0]
for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    p = probs[0, t-1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(t, repr(tok_txt), f"{p:.3e}")

import math
import torch

log_probs = torch.log_softmax(logits, dim=-1)
input_ids = inputs["input_ids"][0]

total_logp = 0.0
n = 0

for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    lp = log_probs[0, t-1, tok_id].item()
    total_logp += lp
    n += 1

avg_neg_logp = -total_logp / n
ppl = math.exp(avg_neg_logp)

print("total_logp:", total_logp)
print("avg_neg_logp:", avg_neg_logp)
print("perplexity:", ppl)


phrases = [
    "Artificial intelligence is fascinating.",
    "Artificial fascinating intelligence is."
]

results = {}
for ph in phrases:
    inputs_ph = tokenizer(ph, return_tensors="pt")
    input_ids_ph = inputs_ph["input_ids"][0]
    
    with torch.no_grad():
        outputs_ph = model(**inputs_ph)
        logits_ph = outputs_ph.logits
    
    log_probs_ph = torch.log_softmax(logits_ph, dim=-1)
    
    total_logp_ph = 0.0
    for t in range(1, len(input_ids_ph)):
        tok_id = input_ids_ph[t].item()
        lp = log_probs_ph[0, t-1, tok_id].item()
        total_logp_ph += lp
    
    n_ph = len(input_ids_ph) - 1
    avg_neg_logp_ph = -total_logp_ph / n_ph
    ppl_ph = math.exp(avg_neg_logp_ph)
    
    results[ph] = ppl_ph
    print(f"\nPhrase: {ph}")
    print(f"  Log-prob totale: {total_logp_ph:.6f}")
    print(f"  Perplexité: {ppl_ph:.4f}")

print(f"\nRapport perplexité (désordonnée/ordonnée): {results[phrases[1]]/results[phrases[0]]:.2f}x")


phrase_fr = "L'intelligence artificielle est fascinante."
inputs_fr = tokenizer(phrase_fr, return_tensors="pt")
input_ids_fr = inputs_fr["input_ids"][0]

with torch.no_grad():
    outputs_fr = model(**inputs_fr)
    logits_fr = outputs_fr.logits

log_probs_fr = torch.log_softmax(logits_fr, dim=-1)

total_logp_fr = 0.0
for t in range(1, len(input_ids_fr)):
    tok_id = input_ids_fr[t].item()
    lp = log_probs_fr[0, t-1, tok_id].item()
    total_logp_fr += lp

n_fr = len(input_ids_fr) - 1
avg_neg_logp_fr = -total_logp_fr / n_fr
ppl_fr = math.exp(avg_neg_logp_fr)

print(f"Phrase FR: {phrase_fr}")
print(f"  Perplexité: {ppl_fr:.4f}")
print(f"\nComparaison:")
print(f"  Phrase EN ordonnée: {results[phrases[0]]:.4f}")
print(f"  Phrase EN désordonnée: {results[phrases[1]]:.4f}")
print(f"  Phrase FR: {ppl_fr:.4f}")

prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits  # (1, seq_len, vocab)

# TODO: récupérer la distribution pour le prochain token (dernier pas de temps)
last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

topk = 10
vals, idx = torch.topk(last_probs, k=topk)

for p, tid in zip(vals.tolist(), idx.tolist()):
    print(repr(tokenizer.decode([tid])), f"{p:.3e}")