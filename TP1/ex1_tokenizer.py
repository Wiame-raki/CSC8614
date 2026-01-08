from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
phrase = "Artificial intelligence is metamorphosing the world!"

# TODO: tokeniser la phrase
tokens = tokenizer.tokenize(phrase)

print(tokens)

# TODO: obtenir les IDs
token_ids = tokenizer.encode(phrase)
print("Token IDs:", token_ids)

print("Détails par token:")
for tid in token_ids:
    # TODO: décoder un seul token id
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))

phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

tokens2 = tokenizer.tokenize(phrase2)
print(tokens2)

long_word_tokens = []
start_idx = None
for i, token in enumerate(tokens2):
    clean_token = token.strip(".,!?;:")  # enlève la ponctuation
    if not clean_token:
        continue
    if "ant" in clean_token.lower() or (start_idx is not None and i <= start_idx + 10):
        if start_idx is None:
            start_idx = i
        long_word_tokens.append(clean_token)

print(f"\nTokens pour 'antidisestablishmentarianism': {long_word_tokens}")
print(f"Nombre de sous-tokens: {len(long_word_tokens)}")