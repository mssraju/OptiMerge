OptiMerge: A Proxy-Guided Multi-Objective Framework for Configuration Optimization in Merging Large Language Models

This repository contains the implementation and experimental resources for the paper:

A Proxy-Guided Multi-Objective Framework for Configuration Optimization in Merging Large Language Models

Overview

OptiMerge is a multi-objective evolutionary framework for automatically discovering effective model-merging configurations for Large Language Models (LLMs).

Instead of manually tuning model-merging parameters or performing an exhaustive grid search, OptiMerge formulates model merging as a multi-objective optimization problem. The framework searches for merging configurations that provide a good balance of performance across multiple downstream tasks.

The framework uses a structured representation of layer-wise merging parameters and a proxy-guided screening strategy to reduce the number of expensive full model evaluations.

In this repository, the main experimental setting focuses on TIES-Merging with Mistral-7B models, where performance on the following three benchmarks is jointly considered:

GSM8K
HumanEval
MedQA

The three task accuracies are treated as optimization objectives.

Main Entry Point

The main executable file is:

Mistral3B.py

Running:

python Mistral3B.py

starts the OptiMerge optimization process.

The program searches for an optimized TIES-Merging configuration using Mistral-7B as the base model and simultaneously considers performance on:

GSM8K
HumanEval
MedQA

as the optimization objectives.

The optimization process searches over the model-merging configuration and evaluates candidate merged models across the three tasks to identify configurations that provide favorable multi-task performance.

Optimization Framework

OptiMerge formulates model merging as a multi-objective optimization problem.

Objectives

The optimization objectives are the task-specific accuracies:

Objective 1: GSM8K accuracy
Objective 2: HumanEval accuracy
Objective 3: MedQA accuracy
Decision Variables

The decision variables represent the parameters controlling the model-merging process, including layer-wise merging parameters.

A structured chromosome representation is used to encode these parameters so that evolutionary operators can operate on the merging configuration.

Evolutionary Optimization

OptiMerge uses NSGA-II to search the configuration space and maintain a diverse set of non-dominated solutions.

The framework additionally employs a Proxy-Guided Screening (PGS) strategy to identify low-potential configurations before expensive full downstream-task evaluation.

TIES Model Merging

The main experiment in this repository uses TIES-Merging.

The optimization process searches for TIES configurations that provide a favorable trade-off between performance on:

GSM8K
HumanEval
MedQA

The TIES merging implementation is integrated with the OptiMerge optimization process.

Experimental Models

The experiments use:

Mistral-7B-Instruct-v0.3
Mathstral-7B-v0.1

as the base models for the model-merging experiments described in the paper.

Running the Main Experiment

After installing the required dependencies and preparing the required models and datasets, run:

python Mistral3B.py

The script initializes the optimization process and searches for promising TIES-Merging configurations.

The resulting configurations can then be evaluated on the target benchmarks to analyze their multi-task performance.

Benchmarks
GSM8K

GSM8K is used to measure mathematical reasoning performance.

HumanEval

HumanEval is used to evaluate code-generation performance.

MedQA

MedQA is used to evaluate medical question-answering performance.

The three benchmarks provide complementary measures of the capabilities of the resulting merged models.

Experimental Configuration

The experiments reported in the paper investigate different chromosome segment lengths, including:

len = 1
len = 4

The population size is set to:

8

The termination criteria are:

len = 1: 32 functional evaluations
len = 4: 64 functional evaluations

The experiments were conducted using an NVIDIA GeForce RTX 3090 GPU.

Repository Structure
OptiMerge-TIES-Mist/
│
├── Mistral3B.py          # Main OptiMerge optimization script
├── Para.py               # Optimization-related implementation
├── Para_v1.py            # Alternative/version of optimization implementation
├── ModelEvaluator.py     # Model evaluation utilities
├── LLMLoad.py            # Model loading utilities
├── LogME.py              # LogME-related utilities
├── utils.py              # General utilities
│
├── yml/                  # Model-merging configuration files
├── data/                 # Dataset and experimental data
│
├── gsm8k/                # GSM8K evaluation resources
├── humaneval/            # HumanEval evaluation resources
├── human_eval/           # HumanEval evaluation implementation
│
├── mergekit/             # Model-merging toolkit
├── model/                # Model-related resources
│
├── setup.py
├── LICENSE
└── README.md
Reproducibility

To reproduce the main optimization experiment:

Install the required Python dependencies.
Prepare the required base models.
Prepare the GSM8K, HumanEval, and MedQA datasets.
Configure the model-merging parameters.
Run:
python Mistral3B.py
Evaluate the resulting merged configurations on the three target benchmarks.

The optimization process is computationally intensive because candidate merged models need to be evaluated on downstream tasks.