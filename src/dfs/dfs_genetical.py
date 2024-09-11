import os
import random
import numpy as np
import logging
from typing import List, Tuple
from deap import base, creator, tools, algorithms
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
import wandb
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_RANDOM_SEED = 666
NEW_MODEL_SIZE = 45

def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Set random seed for all libraries for reproducibility.
    
    Args:
        seed (int): The seed value for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

wandb.login(key=os.environ['WANDB_KEY'])
wandb.init(project="layer_optimization_stacking", name='Genetic for gemma2b-instruct')

table = wandb.Table(columns=["Layers", "Perplexity"])


class LayerStackingOptimization:
    """Class to perform layer stacking optimization using genetic algorithms."""
    
    def __init__(self, cuda_idx: int = 0) -> None:
        """Initialize the LayerStackingOptimization class.
        
        Args:
            cuda_idx (int): CUDA device index to be used for the model.
        """
        self.device_name = f'cuda:{cuda_idx}'
        self.device = torch.device(self.device_name)
        self.tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-Gemma-2B-instruct")

        logging.basicConfig(
            filename='log_iterations.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info('Initialized LayerStackingOptimization class.')

        self.model1 = AutoModelForCausalLM.from_pretrained(
            "Vikhrmodels/Vikhr-Gemma-2B-instruct",
            token=os.environ['HF_TOKEN'],
            device_map="cpu"
        )
        self.model2 = AutoModelForCausalLM.from_pretrained(
            "Vikhrmodels/Vikhr-Gemma-2B-instruct",
            token=os.environ['HF_TOKEN'],
            device_map="cpu"
        )
        self.new_config = AutoConfig.from_pretrained("Vikhrmodels/Vikhr-Gemma-2B-instruct")
        self.new_config.num_hidden_layers = NEW_MODEL_SIZE

        self.num_layers_m1 = self.model1.config.num_hidden_layers
        self.num_layers_m2 = self.model2.config.num_hidden_layers

        dataset = load_dataset('Vikhrmodels/GrandMaster-PRO-MAX', split='test').to_pandas()[['conversation']]
        dataset = dataset[dataset.conversation.apply(len) == 2]
        dataset_shuffled = dataset.sample(2300, random_state=DEFAULT_RANDOM_SEED).sample(frac=1, random_state=DEFAULT_RANDOM_SEED).reset_index(drop=True)
        train, val, test = dataset_shuffled.iloc[:1000], dataset_shuffled.iloc[1000:1300], dataset_shuffled.iloc[1300:2300]
        extract_content = lambda df: df.conversation.apply(lambda x: [x[0]['content'], x[1]['content']])
        train, val, test = map(extract_content, [train, val, test])
        train = train.sample(400, random_state=DEFAULT_RANDOM_SEED)
        self.train_texts = train.map(lambda row: row[0] + " " + row[1]).tolist()

    def create_new_model(self, layer_indices: List[Tuple[int, int]]) -> AutoModelForCausalLM:
        """Create a new model by selecting layers from two pre-trained models.
        
        Args:
            layer_indices (List[Tuple[int, int]]): List of tuples where each tuple contains model index (1 or 2) and layer index.
        
        Returns:
            AutoModelForCausalLM: The newly created model with selected layers.
        """
        logging.info(f"Creating model with layers: {layer_indices}")
        new_model = AutoModelForCausalLM.from_config(self.new_config)
        new_model.embed_tokens = self.model1.base_model.embed_tokens.to(self.device)
        new_model.lm_head = self.model1.lm_head.to(self.device)

        for i, (model_choice, layer_index) in enumerate(layer_indices[:self.new_config.num_hidden_layers]):
            if model_choice == 1 and layer_index >= self.num_layers_m1:
                logging.warning(f"Layer index {layer_index} out of range for model 1. Skipping.")
                continue
            if model_choice == 2 and layer_index >= self.num_layers_m2:
                logging.warning(f"Layer index {layer_index} out of range for model 2. Skipping.")
                continue

            model_layer = (self.model1 if model_choice == 1 else self.model2).base_model.layers[layer_index].to(self.device)
            new_model.base_model.layers[i] = model_layer
            model_layer.to("cpu")
            torch.cuda.empty_cache()

        return new_model

    def evaluate_model(self, layer_indices: List[Tuple[int, int]]) -> Tuple[float]:
        """Evaluate the perplexity of a model created with the specified layers.
        
        Args:
            layer_indices (List[Tuple[int, int]]): List of tuples where each tuple contains model index (1 or 2) and layer index.
        
        Returns:
            Tuple[float]: Perplexity of the model.
        """
        temp_model = self.create_new_model(layer_indices[:self.new_config.num_hidden_layers])
        temp_model.to(self.device)
        temp_model.eval()
        loss_fn = CrossEntropyLoss()
        total_loss = 0
        for test_input in self.train_texts:
            inputs = self.tokenizer(test_input, return_tensors="pt")
            inputs.to(self.device)
            with torch.no_grad():
                outputs = temp_model(**inputs)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()

                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                num_tokens = shift_labels.numel()
                loss = loss / num_tokens
                total_loss += loss.item()

        average_loss = total_loss / len(self.train_texts)
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        logging.info(f"Evaluated naive layer indices: {layer_indices} with perplexity: {perplexity}")

        layer_indices_str = str(layer_indices)    
        table.add_data(layer_indices_str, perplexity)
        wandb.log({"perplexity": perplexity})

        return perplexity,

    def generate_naive_layer_indices(self) -> List[Tuple[int, int]]:
        """Generate a set of naive layer indices for creating a new model.
        
        Returns:
            List[Tuple[int, int]]: List of tuples where each tuple contains model index (1 or 2) and layer index.
        """
        model1_range = list(range(self.num_layers_m1))
        model2_range = list(range(self.num_layers_m2))
        combined_layers = [(1, layer) for layer in model1_range] + [(2, layer) for layer in model2_range]
        counts, selected_layers = Counter(), []

        while len(selected_layers) < self.new_config.num_hidden_layers:
            candidate = random.choice(combined_layers)
            if counts[candidate] < 2:
                selected_layers.append(candidate)
                counts[candidate] += 1

        selected_layers = sorted(selected_layers, key=lambda x: x[1])
        logging.info(f"Generated naive layer indices: {selected_layers}")
        return selected_layers[:self.new_config.num_hidden_layers]

    def evaluate_original_model(self) -> None:
        """Evaluate the perplexity of the original models."""
        original_indices = [(1, i) for i in range(self.num_layers_m1)]
        second_indices = [(2, i) for i in range(self.num_layers_m2)]
    
        perplexity  = self.evaluate_model(original_indices)[0]
        perplexity_ = self.evaluate_model(second_indices)[0]
    
        print(f"Perplexity of the first model: {perplexity}")
        print(f"Perplexity of the second model: {perplexity_}")

        sns.set(style="whitegrid")
        plt.figure(figsize=(6, 4))
        
        colors = ["#4c72b0", "#dd8452"]
        
        labels = ["M1 PPL", "M2 PPL"]
        values = [perplexity, perplexity_]
        
        ax = sns.barplot(x=labels, y=values, palette=colors)
        
        sns.despine(left=True, bottom=True)
        
        plt.title("Bar Plot of M1/M2 Perplexities", fontsize=14, color='white')
        plt.ylabel("Perplexity", fontsize=12, color='white')
        plt.xlabel("", fontsize=12, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        ax.set_facecolor('#333333')
        
        plt.savefig("white_text_bar_plot.png", bbox_inches='tight', transparent=True)
        plt.close()
        
        wandb.log({"M1 / M2 PPLs": wandb.Image("white_text_bar_plot.png")})
        logging.info(f"PPL for original M1: {perplexity}")
        logging.info(f"PPL for original M2: {perplexity_}")
        wandb.log({"original_perplexity M1": perplexity})
        wandb.log({"original_perplexity M2": perplexity_})

    def run_genetic_algorithm(self, population_size: int = 3, generations: int = 100) -> None:
        """Run a genetic algorithm to find the optimal layer configuration.
        
        Args:
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run the algorithm.
        """
        self.evaluate_original_model()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self.generate_naive_layer_indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", self.cx_ordered)
        toolbox.register("mutate", self.mut_shuffle_indexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate_model)

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", min)
        stats.register("avg", lambda x: sum(val[0] for val in x) / len(x) if len(x) > 0 else float('inf'))

        logging.info("Starting genetic algorithm...")
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_perplexity = best_individual.fitness.values[0]
        logging.info(f"Best individual: {best_individual}, Best perplexity: {best_perplexity}")
        # Логирование финальных результатов в wandb
        table_best = wandb.Table(columns=["best_layer_indices", "best_perplexity"])
        table_best.add_data(str(best_individual), best_perplexity)
        wandb.log({"Final best layer and ppl": table_best})
        wandb.log({"Layers and PPL": table})


    @staticmethod
    def cx_ordered(ind1: List[Tuple[int, int]], ind2: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Perform ordered crossover between two individuals.
        
        Args:
            ind1 (List[Tuple[int, int]]): First individual.
            ind2 (List[Tuple[int, int]]): Second individual.
        
        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]: The crossed individuals.
        """
        size = min(len(ind1), len(ind2))
        a, b = sorted(random.sample(range(size), 2))
        temp1 = ind1[a:b] + [item for item in ind2 if item not in ind1[a:b]]
        temp2 = ind2[a:b] + [item for item in ind1 if item not in ind2[a:b]]
        ind1[:], ind2[:] = sorted(temp1), sorted(temp2)
        return ind1, ind2

    @staticmethod
    def mut_shuffle_indexes(individual: List[Tuple[int, int]], indpb: float) -> Tuple[List[Tuple[int, int]]]:
        """Mutate an individual by shuffling its layer indices.
        
        Args:
            individual (List[Tuple[int, int]]): Individual to be mutated.
            indpb (float): Probability of mutation for each index.
        
        Returns:
            Tuple[List[Tuple[int, int]]]: Mutated individual.
        """
        size = len(individual)
        for i in range(size):
            if random.random() < indpb:
                swap_idx = random.randint(0, size - 1)
                individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        individual[:] = sorted(individual)
        return individual,


if __name__ == "__main__":
    optimizer = LayerStackingOptimization()
    optimizer.run_genetic_algorithm(population_size=50, generations=100)
