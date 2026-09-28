import os
# Set CUDA_VISIBLE_DEVICES at the very beginning of the script
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import multiprocessing
import pynvml
import time

import subprocess
from deepeval.benchmarks import GSM8K, HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import logging
import numpy as np
from LLMLoad import LLMLoad
import json
import traceback
from human_eval.data import write_jsonl, read_problems
from vllm import LLM as v_llm #edited by aojun

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init pynvml
pynvml.nvmlInit()

# get gpus num
device_count = pynvml.nvmlDeviceGetCount()

def get_free_gpu():
    """
    检测空闲的GPU，并返回空闲GPU的索引列表
    """
    free_gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_utilization = (memory_info.used / memory_info.total) * 100
        if memory_utilization < 5:  # Assume that the memory utilization is less than 5% indicates free
            free_gpus.append(i)
    return free_gpus

def run_process_on_gpu(process_function, args, gpu_id):
    """
    在指定的GPU上运行进程
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    temp_list = list(args)
    temp_list.append(gpu_id)
    args = tuple(temp_list)
    process = multiprocessing.Process(target=process_function, args=args)
    process.start()
    return process



class Para:
    
    @classmethod
    def extract_full_paths(cls, base_dir_model):
        # List all directories in the base_dir_model and get their full paths with '/' at the end
        full_paths = [os.path.join(base_dir_model, folder) + '/' for folder in os.listdir(base_dir_model) if os.path.isdir(os.path.join(base_dir_model, folder))]
        return full_paths

    @classmethod
    def parallel_evaluation_gsm(cls, base_models, result_file="test_gsm.txt"):
        """
        Evaluate multiple models in parallel using the specified number of GPUs.
        
        Args:
            cls: The class calling this method.
            base_models (list): List of model names to evaluate.
            result_file (str): 
            
        Returns:
            dict: Dictionary with model names and their corresponding accuracies.
        """
        # Check available GPUs
        free_gpus = get_free_gpu()
        allocated_gpus = set()

        # Create the result file
        with open(result_file, 'w') as f:
            pass

        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_on_gsm, (base_model, result_file)))

        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running function '{function.__name__}' with args {args} on GPU {gpu_id}")
                    process = run_process_on_gpu(function, args, gpu_id)
                    break
                else:
                    time.sleep(30)
                    allocated_gpus.clear()

        all_finished = False
        while all_finished is not True:
            has_nums = 0
            time.sleep(60)
            assert os.path.exists(result_file) is True
            with open(result_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
            if has_nums >= len(base_models):
                all_finished = True

        model_accuracies = {}
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(': ')
                model_name = parts[0]
                accuracy = parts[1]
                model_accuracies[model_name] = accuracy

                # Save accuracy to file in respective model directory
                accuracy_file_path = os.path.join(model_name, "accuracy.txt")
                with open(accuracy_file_path, 'w') as accuracy_file:
                    accuracy_file.write(f"GSM Accuracy: {accuracy}\n")
                
        #pynvml.nvmlShutdown()
        time.sleep(60)
        return model_accuracies

    @classmethod
    def evaluate_model_on_gsm(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        """
        Evaluate a single model on a specified GPU and return its accuracy.
        
        Args:
            cls: The class calling this method.
            base_model (str): The name of the model to evaluate.
            result_file (str): 
            gpu_id (int): The ID of the GPU to use.
            
        Returns:
            tuple: Model name and its accuracy.
        """

        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")

        # Define the benchmark
        benchmark = GSM8K(
            n_shots=0,
            enable_cot=True
        )

        # Load the model and tokenizer
        model = v_llm(model=base_model, tensor_parallel_size=1, max_model_len=8192) #edited by aojun
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        LLM = LLMLoad(model=model, tokenizer=tokenizer)

        try:
            # Evaluate the benchmark and get accuracy directly
            accuracy = benchmark.evaluate(model=LLM)
            logger.info(f"Accuracy for {base_model} on GPU {gpu_id}: {accuracy}")
        except Exception as e:
            logger.error(f"Error evaluating model {base_model} on GPU {gpu_id}: {str(e)}")
            accuracy = 0
        finally:
            torch.cuda.empty_cache()

        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {accuracy}\n")

        return base_model, accuracy
    

    @classmethod
    def parallel_evaluation_humaneval(cls, base_models, result_file="test_humaneval.txt"):
        """
        Evaluate multiple models in parallel using the specified number of GPUs.
        
        Args:
            cls: The class calling this method.
            base_models (list): List of model names to evaluate.
            result_file (str): 
            
        Returns:
            dict: Dictionary with model names and their corresponding accuracies.
        """
        # Check available GPUs
        free_gpus = get_free_gpu()
        allocated_gpus = set()

        # Create the result file
        with open(result_file, 'w') as f:
            pass

        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_on_humaneval, (base_model, result_file)))

        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running function '{function.__name__}' with args {args} on GPU {gpu_id}")
                    process = run_process_on_gpu(function, args, gpu_id)
                    break
                else:
                    time.sleep(30)
                    allocated_gpus.clear()

        all_finished = False
        while all_finished is not True:
            has_nums = 0
            time.sleep(60)
            assert os.path.exists(result_file) is True
            with open(result_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
            if has_nums >= len(base_models):
                all_finished = True

        model_accuracies = {}
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(': ')
                model_name = parts[0]
                accuracy = parts[1]
                model_accuracies[model_name] = accuracy
                
                # Open existing accuracy.txt file and append the current accuracy
                accuracy_file_path = os.path.join(model_name, "accuracy.txt")
                if os.path.exists(accuracy_file_path):
                    with open(accuracy_file_path, 'a') as accuracy_file:
                        accuracy_file.write(f"HumanEval Accuracy: {accuracy}\n")
                else:
                    # If the file doesn't exist, create it and add the accuracy
                    with open(accuracy_file_path, 'w') as accuracy_file:
                        accuracy_file.write(f"HumanEval Accuracy: {accuracy}\n")
        #pynvml.nvmlShutdown()
        time.sleep(60)
        return model_accuracies

    @classmethod
    def evaluate_model_on_humaneval(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        """
        Evaluate a single model on a specified GPU and return its accuracy.
        
        Args:
            cls: The class calling this method.
            base_model (str): The name of the model to evaluate.
            result_file (str): Path to the file where results will be logged.
            gpu_id (int): The ID of the GPU to use.
            
        Returns:
            tuple: Model name and its accuracy.
        """
    
        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")
    
        model = v_llm(model=base_model, tensor_parallel_size=1, max_model_len=8192)  # edited by aojun
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        LLM = LLMLoad(model=model, tokenizer=tokenizer)
    
        accuracy = 0  # Initialize accuracy variable
    
        try:
            problems = read_problems()
    
            num_samples_per_task = 10
            samples = [
                dict(task_id=task_id, completion=LLM.generate_samples(problems[task_id]["prompt"], n=1, temperature=0.8))
                for task_id in problems
                for _ in range(num_samples_per_task)
            ]
            
            # Save generated samples
            filename = f"test_{gpu_id}.jsonl"
            write_jsonl(filename, samples)
            
            # Run evaluation
            logger.info(f"Evaluating functional correctness for {filename}")
            from subprocess import check_output  # Using subprocess to capture output
            evaluation_output = check_output(["evaluate_functional_correctness", filename], text=True)
            
            # Extract pass@1 from evaluation output
            import re
            match = re.search(r"'pass@1':\s*([\d.]+)", evaluation_output)
            if match:
                accuracy = float(match.group(1))
            else:
                logger.warning("Failed to extract pass@1 from evaluation output.")
            
        except Exception as e:
            logger.error(f"Error evaluating model {base_model} on GPU {gpu_id}: {str(e)}")
            accuracy = 0
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Deleted temporary file: {filename}")
            else:
                logger.warning(f"Temporary file not found for deletion: {filename}")
            torch.cuda.empty_cache()
    
        # Log results to the result file
        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {accuracy}\n")
    
        return base_model, accuracy
        
    @classmethod
    def parallel_evaluation_medqa(cls, base_models, result_file="test_medqa.txt"):
        """
        Evaluate multiple models in parallel using the specified number of GPUs.
        
        Args:
            cls: The class calling this method.
            base_models (list): List of model names to evaluate.
            result_file (str): 
            
        Returns:
            dict: Dictionary with model names and their corresponding accuracies.
        """
        # Check available GPUs
        free_gpus = get_free_gpu()
        allocated_gpus = set()

        # Create the result file
        with open(result_file, 'w') as f:
            pass

        process_function_calls = []
        for base_model in base_models:
            process_function_calls.append((cls.evaluate_model_on_medqa, (base_model, result_file)))

        for function_call in process_function_calls:
            function, args = function_call
            while True:
                free_gpus = get_free_gpu()
                available_gpus = [gpu for gpu in free_gpus if gpu not in allocated_gpus]
                if available_gpus:
                    gpu_id = available_gpus[0]
                    allocated_gpus.add(gpu_id)
                    print(f"Running function '{function.__name__}' with args {args} on GPU {gpu_id}")
                    process = run_process_on_gpu(function, args, gpu_id)
                    break
                else:
                    time.sleep(30)
                    allocated_gpus.clear()

        all_finished = False
        while all_finished is not True:
            has_nums = 0
            time.sleep(60)
            assert os.path.exists(result_file) is True
            with open(result_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
            if has_nums >= len(base_models):
                all_finished = True

        model_accuracies = {}
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                parts = line.split(': ')
                model_name = parts[0]
                accuracy = parts[1]
                model_accuracies[model_name] = accuracy

                # Open existing accuracy.txt file and append the current accuracy
                accuracy_file_path = os.path.join(model_name, "accuracy.txt")
                if os.path.exists(accuracy_file_path):
                    with open(accuracy_file_path, 'a') as accuracy_file:
                        accuracy_file.write(f"MedQA Accuracy: {accuracy}\n")
                else:
                    # If the file doesn't exist, create it and add the accuracy
                    with open(accuracy_file_path, 'w') as accuracy_file:
                        accuracy_file.write(f"MedQA Accuracy: {accuracy}\n")
                
        #pynvml.nvmlShutdown()
        time.sleep(60)
        return model_accuracies


    @classmethod
    def evaluate_model_on_medqa(cls, base_model: str, result_file: str, gpu_id: int) -> tuple:
        """
        Evaluate a single model on the MedQA dataset and return its accuracy.
    
        Args:
            cls: The class calling this method.
            base_model (str): The name of the model to evaluate.
            result_file (str): Path to the file where results will be logged.
            gpu_id (int): The ID of the GPU to use.
    
        Returns:
            tuple: Model name and its accuracy.
        """
        
        logger.info(f"Evaluating {base_model} on GPU {gpu_id}")
    
        # Load model and tokenizer
        model = v_llm(model=base_model, tensor_parallel_size=1, max_model_len=8192)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        LLM = LLMLoad(model=model, tokenizer=tokenizer)
    
        correct_predictions = 0
        total_questions = 0
        test_file = '/home/sri/RankingBased/Mistral3B/medqa.jsonl'
        
        try:
            # Load MedQA test dataset
            with open(test_file, 'r') as file:
                test_data = [json.loads(line) for line in file]
    
            predictions = {}
            for idx, item in enumerate(test_data):
                question = item["question"]
                options = item["options"]
                true_answer_idx = item["answer_idx"]  # Correct answer's key, e.g., "C"

                # Construct the prompt with question, options, and instruction
                prompt = f"{question}\n"
                for key, option in options.items():
                    prompt += f"{key}: {option}\n"
                prompt += "\nPlease provide your answer in the format: 'Final Answer: [A, B, C, D, E]'.\n"
        
                # Generate the answer
                generated_answer = LLM.generate_samples(prompt, n=1, temperature=0.8)[0].strip()

                
                # Extract the predicted index (option key) from the first line of the generated answer
                predicted_idx = None
                
                # Get the first line of the generated text
                first_line = generated_answer.split("\n")[0].strip()
                
                if "Final Answer:" in first_line:
                    # Extract the part after "Final Answer:"
                    final_answer = first_line.split("Final Answer:")[-1].strip()
                    # Extract only the first character (expected to be the key: A, B, C, D, E)
                    extracted_key = final_answer.split(":")[0].strip().strip("[]")
                    
                    if extracted_key in options.keys():  # Check if it's one of the valid options
                        predicted_idx = extracted_key
                else:
                    print("No 'Final Answer:' found in the first line of the generated answer.")
                
                # If no valid answer is found, log the failure
                if predicted_idx is None:
                    print(f"Failed to extract valid option from generated answer: {generated_answer}")

                    
        
    
                # Evaluate prediction
                if predicted_idx == true_answer_idx:
                    correct_predictions += 1
                    print("Correct")
    
                predictions[f"Q{idx + 1}"] = {
                    "prompt": prompt,
                    "generated_answer": generated_answer,
                    "predicted_idx": predicted_idx,
                    "true_idx": true_answer_idx
                }
                total_questions += 1
    
            # Calculate accuracy
            accuracy = correct_predictions / total_questions if total_questions > 0 else 0.0
    
            
            logger.info(f"Evaluation completed for {base_model}. Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating model {base_model} on MedQA: {str(e)}")
            accuracy = 0
        finally:
            torch.cuda.empty_cache()
    
        # Log results to the result file
        with open(result_file, 'a') as f:
            f.write(f"{base_model}: {accuracy:.4f}\n")
    
        return base_model, accuracy


   