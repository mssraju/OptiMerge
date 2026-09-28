import os
# Set CUDA_VISIBLE_DEVICES at the very beginning of the script
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import multiprocessing
import pynvml
import time
from subprocess import check_output
import re
import subprocess
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pandas as pd
import logging
import numpy as np
import json
import traceback
from LogME import LogME
from sklearn.decomposition import PCA
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Process
import traceback
import gc
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

def get_free_gpu():
    free_gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if memory_info.used / memory_info.total < 0.05:
            free_gpus.append(i)
    return free_gpus

def run_process_on_gpus(process_function, args, gpu_ids):
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))

    # Spawn a new subprocess with GPU restriction by calling a wrapper script
    def target_wrapper(*args):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
        process_function(*args)

    process = multiprocessing.Process(target=target_wrapper, args=args)
    process.start()
    return process


def get_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(model.device)
        with torch.no_grad():
            output = model(**inputs)
            pooled = output.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(pooled.cpu().numpy())
    return embeddings


def compute_sparsity(merged_model_path, base_model_path):
    # Load both models
    merged_model = AutoModel.from_pretrained(merged_model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)

    merged_state = merged_model.state_dict()
    base_state = base_model.state_dict()

    total_l1_norm = 0.0

    for name in merged_state:
        if name in base_state and merged_state[name].dtype == torch.float32:
            delta = merged_state[name] - base_state[name]
            total_l1_norm += torch.norm(delta, p=1).item()

    return total_l1_norm




class ParaLE:
    
    @classmethod
    def extract_full_paths(cls, base_dir_model):
        # List all directories in the base_dir_model and get their full paths with '/' at the end
        full_paths = [os.path.join(base_dir_model, folder) + '/' for folder in os.listdir(base_dir_model) if os.path.isdir(os.path.join(base_dir_model, folder))]
        return full_paths

    
    @classmethod
    def parallel_evaluation_logmegsm(cls, base_models, result_file="LogMEGSM.txt"):
        allocated_gpus = set()
    
        with open(result_file, 'w') as f:
            pass
    
        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_for_logmegsm, (base_model, result_file)))
    
        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running function '{function.__name__}' with args {args} on GPU {gpu_id}")
                    
                    # ? Include gpu_id explicitly
                    process = run_process_on_gpus(function, args + (gpu_id,), gpu_id)
                    
                    break
                else:
                    print("Waiting for a free GPU...")
                    time.sleep(120)
                    allocated_gpus.clear()
    
        # Wait for completion
        all_finished = False
        while not all_finished:
            has_nums = 0
            time.sleep(60)
            with open(result_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
            if has_nums >= len(base_models):
                all_finished = True
    
        model_logme = {}
        with open(result_file, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                model_name, score = parts[0], parts[1]
                model_logme[model_name] = score
    
                score_file_path = os.path.join(model_name, "LogME.txt")
                with open(score_file_path, 'w') as score_file:
                    score_file.write(f"GSM LogME: {score}\n")
    
        time.sleep(60)
        return model_logme

    
    @classmethod
    def evaluate_model_for_logmegsm(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        """
        Evaluate a single model on a specified GPU and return its LogME score.
    
        Args:
            base_model (str): The name of the model to evaluate.
            result_file (str): Result file path.
            gpu_id (int): The GPU to use.
    
        Returns:
            tuple: Model name and LogME score.
        """
        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")
    
        device = torch.device("cuda:0")

    
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
    
        gsm8k = load_dataset("gsm8k", "main", split="train[:500]")
        inputs = [x["question"] for x in gsm8k]
        labels = [float(x["answer"].split("####")[-1].strip().replace(",", "")) for x in gsm8k]
    
        X = get_embeddings(inputs, tokenizer, model)
        y = np.array(labels).reshape(-1, 1)
    
        logme = LogME(regression=True)
        score = logme.fit(np.array(X), y)
    
        print(f"LogME score (regression): {score:.4f}")
    
        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {score}\n")
    
        torch.cuda.empty_cache()
    
        return base_model, score
    
    @classmethod
    def parallel_evaluation_logmehe(cls, base_models, result_file="LogMEHE.txt"):
        allocated_gpus = set()
    
        # Reset the result file
        with open(result_file, 'w') as f:
            pass
    
        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_for_logmehe, (base_model, result_file)))
    
        processes = []
    
        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running '{function.__name__}' with args {args} on GPU {gpu_id}")
                    process = run_process_on_gpus(function, args + (gpu_id,), gpu_id)
                    processes.append(process)
                    break
                else:
                    print("Waiting for a free GPU...")
                    time.sleep(120)
                    allocated_gpus.clear()
    
        # Wait for all processes to complete
        for process in processes:
            process.join()
    
        # Verify completion
        all_finished = False
        while not all_finished:
            completed = 0
            time.sleep(60)
            assert os.path.exists(result_file)
            with open(result_file, 'r') as f:
                completed = sum(1 for line in f if line.strip())
            if completed >= len(base_models):
                all_finished = True
                
                
        model_logme = {}
        with open(result_file, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                model_name, score = parts[0], parts[1]
                model_logme[model_name] = score
    
                score_file_path = os.path.join(model_name, "LogMEHE.txt")
                with open(score_file_path, 'w') as score_file:
                    score_file.write(f"HumanEval LogME: {score}\n")
    

        time.sleep(60)
        return model_logme
    
    @classmethod
    def evaluate_model_for_logmehe(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")
    
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # <-- corrected here
        
        device = torch.device("cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
    
        dataset = load_dataset("openai_humaneval", split="test[:150]")
        texts = [item["prompt"] for item in dataset]
    
        features = get_embeddings(texts, tokenizer, model)
    
        n_samples = len(features)
        features = PCA(n_components=min(32, n_samples)).fit_transform(features)
    
        labels = np.random.uniform(low=0.0, high=1.0, size=(n_samples,))
    
        logme = LogME(regression=True)
        score = logme.fit(features, labels)
    
        print(f"? LogME Score on HumanEval = {score:.4f}")
    
        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {score}\n")
    
        return base_model, score

    @classmethod
    def parallel_evaluation_logmemedqa(cls, base_models, result_file="LogMEMEDQA.txt"):
        allocated_gpus = set()
    
        with open(result_file, 'w') as f:
            pass
    
        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_for_logmemedqa, (base_model, result_file)))
    
        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running function '{function.__name__}' with args {args} on GPU {gpu_id}")
                    
                    # ? Include gpu_id explicitly
                    process = run_process_on_gpus(function, args + (gpu_id,), gpu_id)
                    
                    break
                else:
                    print("Waiting for a free GPU...")
                    time.sleep(120)
                    allocated_gpus.clear()
    
        # Wait for completion
        all_finished = False
        while not all_finished:
            has_nums = 0
            time.sleep(60)
            with open(result_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
            if has_nums >= len(base_models):
                all_finished = True
    
        model_logme = {}
        with open(result_file, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                model_name, score = parts[0], parts[1]
                model_logme[model_name] = score
    
                score_file_path = os.path.join(model_name, "LogMEMEDQA.txt")
                with open(score_file_path, 'w') as score_file:
                    score_file.write(f"MEDQA LogME: {score}\n")
    
        time.sleep(60)
        return model_logme

    @classmethod
    def evaluate_model_for_logmemedqa(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        """
        Evaluate a single model on a specified GPU and return its LogME score on MedQA.
    
        Args:
            base_model (str): Path to model.
            result_file (str): Where to write the scores.
            gpu_id (int): Which GPU to run on.
    
        Returns:
            tuple: (model_name, logme_score)
        """
        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")
    
        device = torch.device("cuda:0")
    
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    
        model = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
    
        # Load MEDQA dataset from JSONL
        with open("/home/sri/RankingBased/Mistral3B/medqa.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
    
        questions = [d["question"] for d in data if "question" in d and "answer" in d]
        answers = [d["answer"] for d in data if "question" in d and "answer" in d]
    
        # If answers are categorical, encode them
        le = LabelEncoder()
        y = le.fit_transform(answers).reshape(-1, 1)
    
        X = get_embeddings(questions, tokenizer, model)
    
        logme = LogME(regression=False)  # assuming MedQA is classification
        score = logme.fit(np.array(X), y)
    
        print(f"LogME score (MedQA): {score:.4f}")
    
        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {score}\n")
    
        torch.cuda.empty_cache()
    
        return base_model, score


    


    @classmethod
    def sequential_evaluation_sparsity(cls, base_models, bm, result_file="Sparsity.txt"):
        """
        Evaluate L1 sparsity (difference) between each model in base_models and the base model (bm).
        Runs one by one (no parallelism).
        
        Args:
            base_models (list): List of model paths to evaluate.
            bm (str): Path to base model.
            result_file (str): File to write output scores.
            
        Returns:
            dict: {model_path: sparsity_score}
        """
    
        with open(result_file, 'w') as f:
            pass  # Clear file
    
        model_sparsity = {}
    
        for model_path in base_models:
            print(f"Running sparsity evaluation for {model_path}")
            sparsity_score = cls.evaluate_model_for_sparsity_onebyone(model_path, bm, result_file)
            model_sparsity[model_path] = sparsity_score
    
            # Save result to model directory
            os.makedirs(model_path, exist_ok=True)
            output_path = os.path.join(model_path, "Sparsity.txt")
            with open(output_path, 'w') as f:
                f.write(f"Sparsity: {sparsity_score}\n")
        time.sleep(60)
        return model_sparsity

       
    @classmethod
    def evaluate_model_for_sparsity_onebyone(cls, base_model: str, bm: str, result_file: str) -> float:
        try:
            merged_model = AutoModel.from_pretrained(base_model, trust_remote_code=True).cpu()
            base_model_obj = AutoModel.from_pretrained(bm, trust_remote_code=True).cpu()
    
            merged_state = merged_model.state_dict()
            base_state = base_model_obj.state_dict()
    
            total_l1 = 0.0
            skipped = 0
    
            for name in merged_state:
                if name in base_state:
                    if merged_state[name].shape != base_state[name].shape:
                        skipped += 1
                        continue
                    if merged_state[name].dtype != torch.float32:
                        skipped += 1
                        continue
                    delta = merged_state[name] - base_state[name]
                    total_l1 += torch.norm(delta, p=1).item()
    
            print(f"? Sparsity (L1 Norm) for {base_model}: {total_l1:.4f} (Skipped: {skipped} layers)")
    
            with open(result_file, 'a') as f:
                f.write(f"{base_model}: {total_l1}\n")
    
            return total_l1
    
        except Exception as e:
            print(f"? Error computing sparsity for {base_model}: {e}")
            traceback.print_exc()
            with open(result_file, 'a') as f:
                f.write(f"{base_model}: ERROR\n")
            return 0.0
    
        finally:
            # Force cleanup
            del merged_model, base_model_obj, merged_state, base_state
            gc.collect()
            torch.cuda.empty_cache()

    @classmethod
    def sequential_evaluation_perplexity(cls, base_models, result_file="Perplexity.txt"):
        """
        Evaluate perplexity for each model (one-by-one, no parallelism) over a set of sentences.
    
        Args:
            base_models (list): List of model directories to evaluate.
            result_file (str): Output file to store perplexity scores.
    
        Returns:
            dict: {model_path: perplexity_score}
        """
        import os
    
        with open(result_file, 'w') as f:
            pass  # Clear file
    
        model_perplexities = {}
    
        for model_path in base_models:
            print(f"🔍 Evaluating perplexity for {model_path}")
            perp_score = cls.evaluate_model_for_perplexity_onebyone(model_path, result_file)
            model_perplexities[model_path] = perp_score
    
            # Save result to model directory
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, "Perplexity.txt"), 'w') as pf:
                pf.write(f"Perplexity: {perp_score}\n")
                
        time.sleep(60)
    
        return model_perplexities

        
    @classmethod
    def evaluate_model_for_perplexity_onebyone(cls, base_model: str, result_file: str) -> float:
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model).eval().cpu()
            tokenizer = AutoTokenizer.from_pretrained(base_model)
    
            test_sentences = [
                "The quick brown fox jumps over the lazy dog.",
                "In 2020, the global pandemic disrupted many industries.",
                "Artificial intelligence is transforming healthcare and education.",
                "The capital of France is Paris.",
                "Quantum computing is an emerging field in computer science.",
                "Photosynthesis allows plants to convert sunlight into energy.",
                "The universe is expanding faster than we previously thought.",
                "He opened the book and began to read aloud to the children.",
                "The derivative of sine is cosine.",
                "Climate change poses a serious threat to biodiversity."
            ]
    
            total_loss = 0.0
            total_tokens = 0
    
            for text in test_sentences:
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    token_count = inputs["input_ids"].size(1)
                    total_loss += loss * token_count
                    total_tokens += token_count
    
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
            print(f"? Perplexity for {base_model}: {perplexity:.4f}")
    
            with open(result_file, 'a') as f:
                f.write(f"{base_model}: {perplexity}\n")
    
            return perplexity
    
        except Exception as e:
            print(f"? Error evaluating {base_model}: {e}")
            traceback.print_exc()
            with open(result_file, 'a') as f:
                f.write(f"{base_model}: ERROR\n")
            return 0.0
    
        finally:
            # Force cleanup
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()



   