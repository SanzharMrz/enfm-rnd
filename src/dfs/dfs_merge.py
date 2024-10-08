import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
from cma.evolution_strategy import CMAEvolutionStrategy
import os 
import random
from datasets import load_dataset
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
import logging
from tqdm import tqdm
import time


DEFAULT_RANDOM_SEED = 666

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(
    filename='evolutionary_stacking.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
seed_everything()


class Multiply(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.alpha)


class EvolutionaryStacking:
    def __init__(
        self, 
        models_list: list,
        tokenizer,
        r: int, 
        M: int, 
        n: int, 
        method: str = "optuna", 
        population_size: int = 50, 
        num_generations: int = 100
    ):
        self.models_list = models_list
        self.tokenizer = tokenizer
        self.r = r
        self.M = M
        self.n = n
        self.I_dim = M * r * n
        self.W_dim = M * r * n + 1
        self.method = method
        self.population_size = population_size
        self.num_generations = num_generations
        
        if method == "cma":
            self.cma_es_I = CMAEvolutionStrategy([1] + [0] * (self.I_dim - 2) + [1], 0.1)
            self.cma_es_W = CMAEvolutionStrategy([0.5] * (self.W_dim * (self.W_dim - 1) // 2), 0.2)
        elif method == "mixed":
            self.I_optimizer = BinaryEvolution(self.I_dim, self.population_size)
            self.cma_es_W = CMAEvolutionStrategy([0.5] * (self.W_dim * (self.W_dim - 1) // 2), 0.2)

    def forward_merged_model(self, merged_model: dict, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = merged_model["embed"](input_ids)
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        rot = merged_model["rotary"](emb, position_ids)
        out = emb
        for idx, layer in enumerate(merged_model["layers"]):
            if idx % 2 == 0:
                if isinstance(out, tuple):
                    out = out[0]
                out = layer(out)
            else:
                out = layer(hidden_states=out, position_embeddings=rot)
        logits = merged_model["lm_head"](out)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def calculate_perplexity(self, model, data: list[str]) -> float:
        total_loss = 0.0
        total_tokens = 0
        logger.info('Start inferencing instructions')
        data_size = len(data)
        for i, (instruction, response) in tqdm(enumerate(data), total=data_size, ncols=80):
            input_text = instruction + response
            inputs = self.tokenizer(input_text, return_tensors="pt")
            labels = inputs.input_ids.clone()
            labels[:, :len(self.tokenizer(instruction).input_ids)] = -100
            with torch.no_grad():
                loss = self.forward_merged_model(model, inputs.input_ids.to(device), labels.to(device))
                total_loss += loss.item() * (labels != -100).sum().item()
                total_tokens += (labels != -100).sum().item()

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{data_size} items processed")

        average_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        logger.info(f"Calculated perplexity: {perplexity}")
        return perplexity

    def W_to_matrix(self, W: np.ndarray, dim: int) -> np.ndarray:
        """
        Converts a flat vector W into a matrix form.
    
        Args:
            W (np.ndarray): Flattened weight matrix.
            dim (int): Dimension of the square matrix.
    
        Returns:
            np.ndarray: Reconstructed square matrix from W.
    
        Raises:
            ValueError: If the length of W is not compatible with the expected size of the upper triangular matrix.
        """
        expected_size = (dim * (dim - 1)) // 2
        if len(W) != expected_size:
            raise ValueError(f"Length of W ({len(W)}) does not match the expected size for an upper triangular matrix of dimension {dim}x{dim}, which should be {expected_size}.")
    
        W_matrix = np.zeros((dim, dim))
        cnt = 0
        for i in range(dim):
            for j in range(dim - i - 1):
                W_matrix[i][j] = W[cnt]
                cnt += 1
        return W_matrix

    def get_ith_layer(self, model, i: int):
        """
        Retrieves the ith layer from a model.

        Args:
            model: The model from which to retrieve the layer.
            i (int): Index of the layer to retrieve.

        Returns:
            The ith layer of the model.
        """
        return list(model.model.layers.children())[i]

    def construct_merged_model(self, I: list[int], W: np.ndarray) -> dict:
        """
        Constructs a merged model from the input models and the weight matrix.

        Args:
            I (list[int]): Binary list indicating the presence of layers.
            W (np.ndarray): Weight matrix.

        Returns:
            dict: A dictionary representing the constructed merged model.
        """
        model = {}
        model["embed"] = list(self.models_list[0].model.children())[0]
        model["layers"] = []
        prev_idx = 0
        for rep in range(self.r):
            for n in range(len(self.models_list)):
                for m in range(self.M):
                    if I[m + n * self.M + rep * self.M * len(self.models_list)] == 1:
                        model["layers"].append(Multiply(W[prev_idx][m + n * self.M + rep * self.M * len(self.models_list)]))
                        prev_idx = m + n * self.M + rep * self.M * len(self.models_list) + 1
                        model["layers"].append(self.get_ith_layer(self.models_list[n], m))
        model["layers"].append(list(self.models_list[0].model.children())[2])
        model["rotary"] = list(self.models_list[0].model.children())[3]
        model["lm_head"] = self.models_list[0].lm_head
        return model
    
    def evaluate_model(self, I: list[int], W: np.ndarray, data) -> float:
        W_matrix = self.W_to_matrix(W, self.W_dim)
        model = self.construct_merged_model(I, W_matrix)
        perplexity = self.calculate_perplexity(model, data)
        logger.info(f"Evaluation perplexity: {perplexity}")
        return perplexity

    def run_mixed_optimization(self, train_data, val_data):
        best_models = []
        for n in tqdm(range(self.num_generations), total=self.num_generations, ncols=80):
            start_time = time.time()
            
            I_list = self.I_optimizer.get_new_population()
            W_list = self.cma_es_W.ask(number=self.population_size)

            for i in tqdm(range(len(I_list)), total=len(I_list), ncols=80):
                metrics = self.evaluate_model(I_list[i], W_list[i], train_data)
                
                if i % 10 == 0:
                    logger.info(f"Evaluating population: {i}/{len(I_list)}")

            logger.info(f"Generation {n} complete, metrics: {metrics}")

            self.I_optimizer.update_population(I_list, metrics)
            self.cma_es_W.tell(W_list, metrics)
            
            val_metrics = [self.evaluate_model(I_list[i], W_list[i], val_data) for i in range(len(I_list))]
            best_idx = np.argmin(val_metrics)
            best_models.append((I_list[best_idx], W_list[best_idx], metrics[best_idx]))
            
            logger.info(f"Generation {n}, best val perplexity: {val_metrics[best_idx]}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generation {n} took {elapsed_time:.2f} seconds")
        
        return best_models


    def run(self, train_data, val_data, n_trials: int = 100):
        """
        Executes the selected optimization method.

        Args:
            train_data: Training data.
            val_data: Validation data.
            n_trials (int): Number of trials for Optuna (if using Optuna). Default is 100.

        Returns:
            The result of the selected optimization method.
        """
        if self.method == "cma":
            return self.run_cma_evolution(train_data, val_data)
        elif self.method == "optuna":
            return self.run_optuna_optimization(train_data, val_data, n_trials)
        elif self.method == "mixed":
            return self.run_mixed_optimization(train_data, val_data)
        else:
            raise ValueError("Invalid method. Choose 'cma', 'mixed', or 'optuna'.")


if __name__ == "__main__":
    dataset = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX", split='test').to_pandas()[['conversation']]
    dataset = dataset[dataset.conversation.apply(len) == 2]
    dataset_shuffled = dataset.sample(2300, random_state=DEFAULT_RANDOM_SEED).sample(frac=1, random_state=DEFAULT_RANDOM_SEED).reset_index(drop=True)
    train, val, test = dataset_shuffled.iloc[:1000], dataset_shuffled.iloc[1000:1300], dataset_shuffled.iloc[1300:2300]
    extract_content = lambda df: df.conversation.apply(lambda x: [x[0]['content'], x[1]['content']])
    train, val, test = map(extract_content, [train, val, test])

    model_name = 'Vikhrmodels/it-5.4-fp16-orpo-v2'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
    )

    models_list = [model, model2]
    M = 32
    r = 1
    n = 2
    n_trials = 100
    method = "mixed"
    
    evolutionary_stacking = EvolutionaryStacking(models_list, tokenizer, r, M, n, method=method)
    best_models = evolutionary_stacking.run(train, val, n_trials)
