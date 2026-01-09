# Rapport TP2 : Fine-tuning GPT-2 for Spam Detection

**Étudiant:** RAKI Wiame
**Date:** Janvier 2026
**OS**: Windows11
**Random Seed:** 123
**Python & Library Versions:**
```
torch: 2.9.1+cpu
tiktoken: 0.12.0
tqdm: 4.67.1
pandas: 2.3.3
matplotlib: 3.10.8
tensorflow: 2.20.0
jupyterlab: 4.5.1
```

## Question 2: Type and Structure of `settings`

- **Type:** Dictionary (`dict`)
- **Keys:** `n_vocab`, `n_ctx`, `n_embd`, `n_head`, `n_layer`
- **Sample Content:**
```
{
  n_vocab: 50257
  n_ctx: 1024
  n_embd: 768
  n_head: 12
  n_layer: 12
}
```

The `settings` dictionary contains the hyperparameters of GPT-2 (124M version).


## Question 3: Type and Structure of `params`

**Answer:**
- **Type:** Dictionary (`dict`)
- **Keys:** `blocks` (list of 12 transformer block weights), `wpe` (position embeddings), `wte` (token embeddings), `g` (final layer norm scale), `b` (final layer norm bias)
- **Structure:** 
  - `params['blocks']`: List of 12 dictionaries (one per layer)
  - Each block contains: `attn` (attention weights), `mlp` (feed-forward weights), `ln_1`, `ln_2` (layer norm parameters)
  - All values are numpy arrays containing the pre-trained weights

The `params` dictionary contains the actual weight matrices from OpenAI's pre-trained GPT-2 model.



## Question 4: Model Config Mapping

The configuration dictionary `cfg` must contain the following keys:

```python
cfg = {
    "vocab_size",      # vocabulary size
    "emb_dim",         # embedding dimension
    "context_length",  # max sequence length
    "drop_rate",       # dropout rate
    "n_layers",        # number of transformer blocks
    "n_heads",         # number of attention heads
    "qkv_bias"         # whether QKV projections use bias
}
```

* The `settings` variable from GPT-2 does not match this structure directly.


## Question 5.1: Shuffle Purpose

`df.sample(frac=1, random_state=123)` shuffles the dataframe to avoid bias in train/test split.

- Ensures spam/ham are distributed uniformly across train/test sets
- Prevents the model from seeing all spam messages in one group
- `random_state=123` guarantees reproducibility
- Without shuffling, sequential spam could bias results

## Question 5.2: Class Balance Analysis

**Observation:**

| Class | Count | Percentage |
|-|-|--|
| Ham   | 3860  | 86.61%    |
| Spam  | 597   | 13.39%    |

**Ratio:** 6.47 (ham to spam)

**Assessment:** HIGHLY UNBALANCED

**Potential Issues:**
1. **Majority Class Bias:** Model learns to predict "ham" for everything
2. **Poor Spam Detection:** Low recall for minority class
3. **Misleading Accuracy:** 87% accuracy by predicting all "ham"
4. **Overfitting to Ham:** Limited diversity in spam examples

**Mitigation:**
- Used **class weights** in loss function: `pos_weight ≈ 6.47`
- Weighted loss penalizes spam misclassification more heavily
- Reported both global and spam-specific accuracy


## Question 7: Batch Count

**Calculation:**
- Training dataset size: **4,457 samples**
- Batch size: **16**
- Number of batches per epoch: **279** (ceiling division: (4457 + 15) // 16)

**Note:** No subsampling applied.

## Question 8: Why Freeze Internal Layers?

**Transfer Learning Principle:**
1. **Reuse Knowledge:** Pre-trained weights capture general language features (syntax, semantics)
2. **Computational Efficiency:** Training only 2×768 + 2 = 1,538 parameters vs 768×768 = 589,824
3. **Data Efficiency:** With 4,457 training samples, freezing prevents overfitting
4. **Stable Training:** Large learning rates won't destroy useful pre-trained knowledge
5. **Transfer:** Learned representations for general text transfer to spam detection task


## Question 10: Training Results & Loss Trend

The training loss is very high and unstable at the beginning of Epoch 1, then steadily decreases, indicating that the model starts learning meaningful patterns.
In Epoch 2, the loss remains moderate but the model collapses into predicting only the minority class (spam), a known effect of strong class weighting.
By Epoch 3, the loss stabilizes at lower values and both training and test accuracies improve significantly.
The close alignment between train and test performance in the final epoch indicates good generalization.

Overall, despite early instability, the model is clearly learning and converging.

