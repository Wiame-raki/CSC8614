# Lab Report - Parameter-Efficient Fine-Tuning with LoRA

**Date:** January 2026  
**Student:** RAKI Wiame 
**Random Seed:** 42  
**Python Version:** 3.12.6 
**Key Libraries:**
- torch==2.9.1
- tensorflow==2.20.0
- tiktoken==0.12.0
- pandas==2.3.3


## Question 1: Structural Modifications Induced by LoRA Injection

A clear structural modification is observed following the injection of LoRA modules. In the *Model Structure After LoRA*, the original `nn.Linear` layers within the transformer blocks are replaced by `LinearWithLoRA` modules. These modules encapsulate the frozen base linear layer and introduce an additional trainable low-rank adaptation branch (`LoRALayer`). Consequently, each transformer block contains six LoRA-wrapped linear components: the query, key, and value projections, the output projection, and the two feed-forward network layers. The final output head remains a standard linear layer, in accordance with the design principles of LoRA-based fine-tuning.



## Question 2: Initial Trainable Parameter Count (LoRA Only)

**Before Adding the Classification Head:**

* Trainable parameters: 1,327,104
* Total parameters: 164,364,288
* Trainable fraction: **0.81%**

With a LoRA rank of 8 and a scaling factor (α) of 16, approximately 99.19% of the model parameters remain frozen during fine-tuning. Only 0.81% of the parameters are updated through the low-rank decomposition matrices. This highlights the effectiveness of LoRA as a parameter-efficient fine-tuning strategy, enabling adaptation with a minimal number of trainable parameters.



## Question 3: Parameter Count After Adding the Classification Head

**After Adding the Classification Head:**

* Trainable parameters: 1,328,642
* Total parameters: 125,768,450
* Trainable fraction: **1.06%**

**Analysis:**
The number of trainable parameters increases marginally by 1,538, corresponding to the weights and biases of the newly introduced two-class classification head. In contrast, the total parameter count decreases substantially—from approximately 164 million to 125 million—due to the replacement of the original language modeling head (which projected to a 50,257-token vocabulary) with a significantly smaller binary classification head. Despite this architectural change, LoRA remains the primary adaptation mechanism, accounting for the majority of trainable parameters at a final proportion of 1.06%.



## Question 4: Training Loss and Accuracy Dynamics

The training process exhibits strong convergence characteristics:

* Initial loss (batch 0): 2.9491
* Loss at batch 10: 1.3096
* Minimum loss: 0.0016 (batch 90)
* Average loss over the epoch: 0.2789
* **Final training accuracy:** 92.79%
```
Epoch 1 | Batch 0 | Loss: 2.9491
Epoch 1 | Batch 10 | Loss: 1.3096
Epoch 1 | Batch 20 | Loss: 0.2150
Epoch 1 | Batch 30 | Loss: 0.3128
Epoch 1 | Batch 40 | Loss: 0.0593
Epoch 1 | Batch 50 | Loss: 0.0281
Epoch 1 | Batch 60 | Loss: 0.0237
Epoch 1 | Batch 70 | Loss: 0.0116
Epoch 1 | Batch 80 | Loss: 0.0032
Epoch 1 | Batch 90 | Loss: 0.0016
Epoch 1 | Batch 100 | Loss: 0.0043
Epoch 1 | Batch 110 | Loss: 0.0035
Epoch 1 | Batch 120 | Loss: 0.2439
Epoch 1 | Batch 130 | Loss: 0.3970
Epoch 1 | Batch 140 | Loss: 0.0324
Epoch 1 Finished | Avg Loss: 0.2789 | Acc: 92.79% | Time: 1983.50s
```
The training loss decreases rapidly during the early stages and stabilizes at a low average value by the end of the epoch. This behavior indicates that the model effectively learns the spam classification task despite updating only 1.06% of its parameters. The sharp convergence underscores the capacity of LoRA’s low-rank adaptations to exploit the representational power of the pre-trained GPT-2 model for downstream classification tasks.



## Question 5: Test Performance and Generalization Capability

**Test accuracy:** **97.66%**

The achieved test accuracy exceeds the training accuracy, indicating strong generalization performance. The LoRA fine-tuned model demonstrates a robust ability to discriminate between spam and non-spam messages on unseen data. This result suggests that LoRA introduces an implicit regularization effect—stemming from low-rank constraints and frozen base parameters—which mitigates overfitting and facilitates the learning of task-relevant features that generalize beyond the training dataset.




