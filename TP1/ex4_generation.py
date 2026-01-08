import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42 # TODO
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50 ,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

#ajout pénalité
def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

#réduction de temp
def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

#augmentation de temp
def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=2.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)


print("5 beam")
import time

start_time = time.time()  # start timer

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

end_time = time.time()  # end timer

txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)

# print execution time
print(f"Generation took {end_time - start_time:.4f} seconds")

print("10 beam")
import time

start_time = time.time()  # start timer

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=10,
    early_stopping=True
)

end_time = time.time()  # end timer

txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)

# print execution time
print(f"Generation took {end_time - start_time:.4f} seconds")

print("20 beam")
import time

start_time = time.time()  # start timer

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=20,
    early_stopping=True
)

end_time = time.time()  # end timer

txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)

# print execution time
print(f"Generation took {end_time - start_time:.4f} seconds")

print("30 beam")
import time

start_time = time.time()  # start timer

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=30,
    early_stopping=True
)

end_time = time.time()  # end timer

txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)

# print execution time
print(f"Generation took {end_time - start_time:.4f} seconds")
       