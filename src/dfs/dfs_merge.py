import torch
import torch.nn as nn
import numpy as np
from cma.evolution_strategy import CMAEvolutionStrategy

class Multiply(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        x = torch.mul(x, self.alpha)
        return x

def calculate_perplexity(model, tokenizer, texts):
    ## TODO: rewrite, not tested
    losses = []
    for text in texts:
        data = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.forward(input_ids=data["input_ids"], labels=data["input_ids"].clone())
            losses.append(out.loss)
    return torch.exp(losses.mean()).item()

def W_to_matrix(W, dim):
    W_matrix = np.zeros((dim, dim))
    cnt = 0
    for i in range(dim):
        for j in range(dim - i - 1):
            W_matrix[i][j] = W[cnt]
            cnt += 1
    return W_matrix

def get_ith_layer(model, i):
    # gets the ith layer from mistral-based model
    return list(model.model.layers.children())[i]

def construct_merged_model(models_list, I, W, r, M):
    model = {}
    model["embed"] = list(models_list[0].model.children())[0]
    model["layers"] = []
    prev_idx = 0
    for rep in range(r):
        for n in range(len(models_list)):
            for m in range(M):
                if I[m + n * M + rep * M * len(models_list)] == 1:
                    model["layers"].append(Multiply(W[prev_idx][m + n * M + rep * M * len(models_list)]))
                    prev_idx = m + n * M + rep * M * len(models_list) + 1
                    model["layers"].append(get_ith_layer(models_list[n], m))
    model["layers"].append(list(models_list[0].model.children())[2])
    model["rotary"] = list(models_list[0].model.children())[3]
    model["lm_head"] = models_list[0].lm_head
    return model

def evaluate_model(I, W, W_dim, data):
    W_matrix = W_to_matrix(W, W_dim)
    model = construct_merged_model(I, W_matrix)
    return calculate_perplexity(model, data)

def run_evolution(cma_es_I, cma_es_W, W_dim, train_data, val_data, population_size, num_generations):
    best_models = []
    for n in range(num_generations):
        I_list = cma_es_I.ask(number=population_size)
        W_list = cma_es_W.ask(number=population_size)
        metrics = [evaluate_model(I_list[i], W_list[i], W_dim, train_data) for i in range(len(I_list))]
        cma_es_I.tell(I_list, metrics)
        cma_es_W.tell(W_list, metrics)
        val_metrics = [evaluate_model(I_list[i], W_list[i], W_dim, val_data) for i in range(len(I_list))]
        best_idx = np.argmin(val_metrics)
        best_models.append((I_list[best_idx], W_list[best_idx], metrics[best_idx]))
        print(f"Generation {n}, best val perplexity: {val_metrics[best_idx]}")
    return best_models

if __name__ == "__main__":
    M = 32
    r = 3
    n = 3
    cma_es_I = CMAEvolutionStrategy([1] + [0] * (M * r * n - 2) + [1], 0.1)
    cma_es_W = CMAEvolutionStrategy([0.5] * ((M * r * n + 2) * (M * r * n + 1) // 2), 0.2)    # M * r * n * (M * r * n + 1) / 2 - size of flattened  flipped upper-triangular matrix