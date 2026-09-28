####################################### Libraries ###########################################
import os,time
# Set CUDA_VISIBLE_DEVICES at the very beginning of the script
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from utils import Utils
import random
from Para_v1 import Para
from Para import ParaLE
import numpy as np
import shutil
import pickle
import yaml
from ModelEvaluator import ModelEvaluator
import pynvml
import gc
import torch
import multiprocessing

def run_in_subprocess(target_func, *args):
    """Run any function in a subprocess to isolate memory (especially CUDA)."""
    p = multiprocessing.Process(target=target_func, args=args)
    p.start()
    p.join()

def run_all_sequential_evaluations(folder_names, base_model):
    try:
        sparsity = ParaLE.sequential_evaluation_sparsity(folder_names, base_model)
        print("? Sparsity Done:", sparsity)

        perplexity = ParaLE.sequential_evaluation_perplexity(folder_names)
        print("? Perplexity Done:", perplexity)
    except Exception as e:
        print("? Error in sequential evaluations:", e)

##############################################################################################
def wait_for_gpu_with_min_free_memory(min_free_gb=5, check_interval=30):
    required_free_bytes = min_free_gb * 1024**3
    pynvml.nvmlInit()
    try:
        while True:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gb = mem_info.free / 1024**3
                if mem_info.free >= required_free_bytes:
                    print(f"[INFO] GPU {i} has {free_gb:.2f} GB free. Proceeding...")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
                    return
            print("[INFO] No GPU with >= 5GB free. Waiting...")
            time.sleep(check_interval)
    finally:
        pynvml.nvmlShutdown()

#################################### Intialization ###########################################
pop_size = 8 # Population Size
max_iter = 10  # Maximum Number of Iterations
var_len  = 1  # Number of Weights and Densities
lower=np.zeros(var_len)
upper=np.ones(var_len)
it       = 0  # Generation 0
FE       = 8
maxFE    = 50
start    = time.time()
base_dir = '/data/llm/sri/Baseline/OptiMerge_TA_Mist_1/'
base_dir_yml = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/yml'
base_dir_model = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/model'
###################################### Main Code ##############################################
Utils.create_directory(base_dir,it,yml_dir_name='yml', models_dir_name='model')
#params = Utils.get_init_params()
################################## Initial Model ##############################################
Initial_model=['mistralai/Mistral-7B-Instruct-v0.3','mistralai/Mathstral-7B-v0.1']
base_model = Initial_model[0]#[Initial_model[0]] * (pop_size // 2) + [Initial_model[1]] * (pop_size // 2)
############################# Initialize Weights and Densities ###############################
weight = Utils.generate_weight_sets(pop_size,var_len)
density = Utils.generate_density_sets(pop_size,var_len)
# ##################### Creating Merging yml files for Merging with TIES #######################

Utils.generate_yml_scripts(
    weight,                       # Positional argument 1
    density,                      # Positional argument 2
    Initial_model,                # Positional argument 3 (models)
    base_model,                   # Positional argument 4 (base_model)
    it,                           # Positional argument 5 (iteration_name)
    output_dir='/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/yml'  # Keyword argument (output_dir)
)
########################### Generating Model with yml files ##################################

tt=0
wait_for_gpu_with_min_free_memory()
Utils.run_mergekit(tt,base_dir_yml, base_dir_model)
folder_names = Para.extract_full_paths(base_dir_model)
############################ Evaluating ######################################################
gsm_Acc=Para.parallel_evaluation_gsm(folder_names)
print(gsm_Acc)
he_Acc=Para.parallel_evaluation_humaneval(folder_names)
print(he_Acc)
medqa_Acc=Para.parallel_evaluation_medqa(folder_names)
print(medqa_Acc)

LogME=ParaLE.parallel_evaluation_logmegsm(folder_names)
print(LogME)
LogHE=ParaLE.parallel_evaluation_logmehe(folder_names)
print(LogHE)
LogMEDQA=ParaLE.parallel_evaluation_logmemedqa(folder_names)
print(LogMEDQA)
run_in_subprocess(run_all_sequential_evaluations, folder_names, base_model)

#################
evaluator = ModelEvaluator(base_dir_model)
# Run evaluation
ranked_indices, composite_scores = evaluator.evaluate()
# Print results
print("Ranked Models (Best to Worst):")
for rank, idx in enumerate(ranked_indices, start=1):
    print(f"Rank {rank}: Model {idx} (Score: {composite_scores[idx]:.3f})") 

############################## Main Loop ######################################################
############################### Data Splitting ##############################################
# Splitting the data into a list of tuples (path, value) directly from dictionary
split_data = list(gsm_Acc.items())  # Convert dictionary items to list of tuples

# Sorting the data by the directory name in ascending order
sorted_data = sorted(split_data, key=lambda x: x[0])

# Initialize numpy array for transformed values
gsm = np.zeros(len(sorted_data))
# Populate the array with 1 - value from sorted data
for i, (_, value) in enumerate(sorted_data):
    gsm[i] = value
gsm=1-gsm
# Splitting the data into a list of tuples (path, value)
split_data = list(he_Acc.items())  # Convert dictionary items to list of tuples

# Sorting the data by the directory name in ascending order
sorted_data = sorted(split_data, key=lambda x: x[0])

# Initialize numpy array for transformed values
he = np.zeros(len(sorted_data))  
# Populate the array with 1 - value from sorted data
for i, (_, value) in enumerate(sorted_data):
    he[i] = value
he=1-he  

# Splitting the data into a list of tuples (path, value)
split_data = list(medqa_Acc.items())  # Convert dictionary items to list of tuples

# Sorting the data by the directory name in ascending order
sorted_data = sorted(split_data, key=lambda x: x[0])

# Initialize numpy array for transformed values
medqa = np.zeros(len(sorted_data))  
# Populate the array with 1 - value from sorted data
for i, (_, value) in enumerate(sorted_data):
    medqa[i] = value
medqa=1-medqa  

#################################################################################
Pop=np.column_stack((gsm, he, medqa))

###################################################################################################
###############################################################################################
while FE < maxFE:
    print('Iteration:',it)
    ###################################################################################################
    mating=Utils.mating(Pop)
    sol_ind=np.array([np.random.choice(row) for row in mating])
    Offw1=[]
    Offw2=[]
    Offd1=[]
    Offd2=[]
    new_weight=[]
    new_density=[]
    for i,value in mating:
        off=Utils.variation(np.vstack([weight[i][0],weight[sol_ind[i]][0]]),lower,upper)
        Offw1.append(off[0])
        off=Utils.variation(np.vstack([weight[i][1],weight[sol_ind[i]][1]]),lower,upper)
        Offw2.append(off[0])
        off=Utils.variation(np.vstack([density[i][0],density[sol_ind[i]][0]]),lower,upper)
        Offd1.append(off[0])
        off=Utils.variation(np.vstack([density[i][1],density[sol_ind[i]][1]]),lower,upper)
        Offd2.append(off[0])
    
    new_weight=[(o1, o2) for o1, o2 in zip(Offw1, Offw2)]
    new_density=[(o1, o2) for o1, o2 in zip(Offd1, Offd2)]
    new_weight=[(np.round(arr1, 2), np.round(arr2, 2)) for arr1, arr2 in new_weight]
    new_density=[(np.round(arr1, 2), np.round(arr2, 2)) for arr1, arr2 in new_density]
    print('New_Weight:',new_weight)
    print('New_Density:',new_density)
    ####################################################################################################
    Utils.create_directory(base_dir,it,yml_dir_name='off_yml', models_dir_name='off_model')
    ##################### Creating Merging yml files for Merging with TIES #############################
    Utils.generate_yml_scripts(
    new_weight,                       # Positional argument 1
    new_density,                      # Positional argument 2
    Initial_model,                # Positional argument 3 (models)
    base_model,                   # Positional argument 4 (base_model)
    tt,                           # Positional argument 5 (iteration_name)
    output_dir='/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_yml'  # Keyword argument (output_dir)
    )
    ########################### Generating Model with yml files #########################################
    base_dir_model = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_model'
    base_dir_yml = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_yml'
    wait_for_gpu_with_min_free_memory()
    Utils.run_mergekit(tt,base_dir_yml, base_dir_model)
    folder_names = Para.extract_full_paths(base_dir_model)
    ############################ Ranking Based ##########################################################
    LogME=ParaLE.parallel_evaluation_logmegsm(folder_names)
    print(LogME)
    LogHE=ParaLE.parallel_evaluation_logmehe(folder_names)
    print(LogHE)
    LogMEDQA=ParaLE.parallel_evaluation_logmemedqa(folder_names)
    print(LogMEDQA)
    run_in_subprocess(run_all_sequential_evaluations, folder_names, base_model)
    
    #################
    evaluator = ModelEvaluator(base_dir_model)
    # Run evaluation
    ranked_indices, composite_scores = evaluator.evaluate()
    # Print results
    print("Ranked Models (Best to Worst):")
    for rank, idx in enumerate(ranked_indices, start=1):
        print(f"Rank {rank}: Model {idx} (Score: {composite_scores[idx]:.3f})")    
        
    threshold = 0.5  # Adjust this value based on your requirement
    # Delete models below threshold
    for idx, score in enumerate(composite_scores):
        model_path = os.path.join(base_dir_model, f"individual_0_{idx:02d}")
        if score < threshold:
            if os.path.exists(model_path):
                print(f"Deleting Model {idx} (Score: {score:.3f}) - below threshold.")
                shutil.rmtree(model_path)
            else:
                print(f"Model {idx} not found at {model_path}, skipping deletion.")
                
                
    # === Step: Rename remaining models sequentially ===
    remaining_folders = sorted([
        f for f in os.listdir(base_dir_model)
        if os.path.isdir(os.path.join(base_dir_model, f)) and f.startswith("individual_0_")
    ])
    
    for new_idx, folder in enumerate(remaining_folders):
        old_path = os.path.join(base_dir_model, folder)
        new_folder_name = f"individual_0_{new_idx:02d}"
        new_path = os.path.join(base_dir_model, new_folder_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed '{folder}' to '{new_folder_name}'")

    ############################# Evaluating ############################################################
    folder_names = Para.extract_full_paths(base_dir_model)
    if len(folder_names) >= 1:
        FE = FE+len(folder_names)
        print('FE:',FE)
        off_gsm_Acc=Para.parallel_evaluation_gsm(folder_names)
        print(off_gsm_Acc)
        off_he_Acc=Para.parallel_evaluation_humaneval(folder_names)
        print(off_he_Acc)
        off_medqa_Acc=Para.parallel_evaluation_medqa(folder_names)
        print(off_medqa_Acc)
        #####################################################################################################
        ############################### Data Splitting ##############################################
        # Splitting the data into a list of tuples (path, value) directly from dictionary
        split_data = list(off_gsm_Acc.items())  # Convert dictionary items to list of tuples
        
        # Sorting the data by the directory name in ascending order
        sorted_data = sorted(split_data, key=lambda x: x[0])
        
        # Initialize numpy array for transformed values
        off_gsm = np.zeros(len(sorted_data))
        # Populate the array with 1 - value from sorted data
        for i, (_, value) in enumerate(sorted_data):
            off_gsm[i] = value
        off_gsm=1-off_gsm
        # Splitting the data into a list of tuples (path, value)
        split_data = list(off_he_Acc.items())  # Convert dictionary items to list of tuples
        
        # Sorting the data by the directory name in ascending order
        sorted_data = sorted(split_data, key=lambda x: x[0])
        
        # Initialize numpy array for transformed values
        off_he = np.zeros(len(sorted_data))  
        # Populate the array with 1 - value from sorted data
        for i, (_, value) in enumerate(sorted_data):
            off_he[i] = value
        off_he= 1-off_he    
        # Splitting the data into a list of tuples (path, value)
        split_data = list(off_medqa_Acc.items())  # Convert dictionary items to list of tuples
        
        # Sorting the data by the directory name in ascending order
        sorted_data = sorted(split_data, key=lambda x: x[0])
        
        # Initialize numpy array for transformed values
        off_medqa = np.zeros(len(sorted_data))  
        # Populate the array with 1 - value from sorted data
        for i, (_, value) in enumerate(sorted_data):
            off_medqa[i] = value
        off_medqa= 1-off_medqa   
        #############################  Environmental Selection ############################################    
        Off_Pop=np.column_stack((off_gsm, off_he, off_medqa))
        Merged_Pop = np.vstack((Pop, Off_Pop))
        print('Merged Pop:',Merged_Pop)
        Merged_Weight = weight+new_weight
        print('Merged Weight:',Merged_Weight)
        Merged_Density=density+new_density
        print('Merged Density:',Merged_Density)
        ########################################################################################################
        [FNo,MNo]=Utils.T_ENS(Merged_Pop, pop_size)
        # Step 1: Select individuals with front number < MaxFNo
        Next = FNo < MNo
        
        # Step 2: If not enough selected, pick from last front
        count = int(np.sum(Next))
        if count < pop_size:
            Last = np.where(FNo == MNo)[0]
            CD = Utils.calculate_crowding_distance(Merged_Pop, FNo)
            Rank = np.argsort(-CD[Last])
            required = pop_size - count
            selected_from_last = Last[Rank[:required]]
            Next[selected_from_last] = True
        
        # Step 3: Enforce exact pop_size by truncating extras (safety net)
        if np.sum(Next) > pop_size:
            excess = np.where(Next)[0][pop_size:]  # drop from end of True list
            Next[excess] = False
        
        final_indices = np.where(Next)[0]

        print("Selected:", final_indices)
        print("Total Selected:", len(final_indices))
        ########################################################################################################
        index=np.where(Next==False)[0]
        folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/model'
        off_folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_model'
        for i in index:
            if i <= pop_size - 1:
                filepath = f"individual_0_0{i}"
                file_path = os.path.join(folder_path, filepath)
                print(file_path)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete the directory and all its contents
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                filepath = f"individual_0_0{i-pop_size}"
                file_path = os.path.join(off_folder_path, filepath)
                print(file_path)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete the directory and all its contents
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        temp_folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/Temp'
        ####################################################################################################
        # Ensure the temp folder exists, create if not
        if not os.path.exists(temp_folder_path):
            os.makedirs(temp_folder_path)
        # Move and rename folders from folder_path to Temp
        next_index = Utils.move_and_rename_folders(folder_path, temp_folder_path)
        # Continue with off_folder_path using the next index from the previous operation
        Utils.move_and_rename_folders(off_folder_path, temp_folder_path, start_index=next_index)
        #####################################################################################################
        # Define the paths
        base_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1'
        temp_folder_path = os.path.join(base_path, 'Temp')
        new_model_path = os.path.join(base_path, 'model')
        
        # Check if the new model path already exists
        if os.path.exists(new_model_path):
            # Optionally back up the existing 'model' directory
            backup_model_path = os.path.join(base_path, 'model_backup')
            if os.path.exists(backup_model_path):
                shutil.rmtree(backup_model_path)  # Remove existing backup if it exists
            shutil.move(new_model_path, backup_model_path)
            print(f"Existing 'model' directory backed up to 'model_backup'")
        
        # Rename 'Temp' to 'model'
        shutil.move(temp_folder_path, new_model_path)
        ########################################################################################################
        # Specify the directory from which to delete files
        folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/yml'
        # List all files in the directory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Unlink (delete) the file or link
                elif os.path.isdir(file_path):
                    continue  # Optionally, you can add code to handle directories here
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
            # Specify the directory from which to delete files
            folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_yml'
            # List all files in the directory
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Unlink (delete) the file or link
                    elif os.path.isdir(file_path):
                        continue  # Optionally, you can add code to handle directories here
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        ########################################################## Sorting Pop #################################
        # Define the folder path containing the model directories
        folder_path = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/model'
        
        # List to store the GSM, HumanEval, and MMLU accuracies for each model
        accuracies = []
        
        # Iterate over each directory inside the folder path in sorted order
        for model_folder in sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1])):
            model_path = os.path.join(folder_path, model_folder)
        
            # Check if it's a directory (model)
            if os.path.isdir(model_path):
                accuracy_file_path = os.path.join(model_path, 'accuracy.txt')
        
                # Check if accuracy.txt exists in the model folder
                if os.path.exists(accuracy_file_path):
                    with open(accuracy_file_path, 'r') as accuracy_file:
                        gsm_accuracy = "N/A"
                        humaneval_accuracy = "N/A"
        
                        # Read the accuracy.txt file line by line to extract accuracies
                        for line in accuracy_file:
                            if "GSM Accuracy" in line:
                                gsm_accuracy = float(line.split(": ")[1].strip())
                            elif "HumanEval Accuracy" in line:
                                humaneval_accuracy = float(line.split(": ")[1].strip())
                            elif "MedQA Accuracy" in line:
                                medqa_accuracy = float(line.split(": ")[1].strip())
        
                        # Store the GSM, HumanEval, and MMLU accuracies in the list
                        accuracies.append([1-gsm_accuracy, 1-humaneval_accuracy, 1-medqa_accuracy])
        
        # Convert the list of accuracies to a numpy array for desired format
        Pop = np.array(accuracies)
        print('Pop:', Pop)
        
        directory_path = "/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/model"
        
        # List to store all weight and density tuples
        all_weights = []
        all_densities = []
        
        # Iterate through each folder in sorted order
        for folder_name in sorted(os.listdir(directory_path), key=lambda x: int(x.split('_')[-1])):
            folder_path = os.path.join(directory_path, folder_name)
            if os.path.isdir(folder_path):
                yaml_file_path = os.path.join(folder_path, 'mergekit_config.yml')
                
                # Check if the YAML file exists in the folder
                if os.path.exists(yaml_file_path):
                    # Read the YAML file
                    with open(yaml_file_path, 'r') as yaml_file:
                        yaml_content = yaml.safe_load(yaml_file)
                        
                        # Extract the weight and density values
                        weights = []
                        densities = []
                        
                        def extract_weights(data):
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if key == 'weight':
                                        weights.append(value)
                                    else:
                                        extract_weights(value)
                            elif isinstance(data, list):
                                for item in data:
                                    extract_weights(item)
                        
                        def extract_densities(data):
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if key == 'density':
                                        densities.append(value)
                                    else:
                                        extract_densities(value)
                            elif isinstance(data, list):
                                for item in data:
                                    extract_densities(item)
                        
                        extract_weights(yaml_content)
                        extract_densities(yaml_content)
                        
                        # Store the weight values in the required format
                        if len(weights) >= 2:
                            weight_1 = np.array(weights[0])
                            weight_2 = np.array(weights[1])
                            all_weights.append((weight_1, weight_2))
                        # Store the density values in the required format
                        if len(densities) >= 2:
                            density_1 = np.array(densities[0])
                            density_2 = np.array(densities[1])
                            all_densities.append((density_1, density_2))
        weight=[]
        density=[]
        weight=all_weights
        density=all_densities
        file_name = f"Info_{it}.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump((weight, ), file)
    it=it+1
    base_dir_yml = '/data/llm/sri/Baseline/OptiMerge_TIES_Mist_1/off_yml'

    for file_name in os.listdir(base_dir_yml):
        file_path = os.path.join(base_dir_yml, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
