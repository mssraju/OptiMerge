import configparser
import numpy as np
import yaml
import os
import subprocess
import shutil

class Utils:
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('settings', 'pop_size')
        params['num_iteration'] = cls.get_params('settings', 'num_iteration')
        return params

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)
    
    @classmethod
    def generate_weight_sets(cls, num_sets, var_len):
        """
        Generate weight sets for TIES grid search (6x6 = 36 combinations).
        Each set contains two arrays (for two models), where set2 = 1 - set1.
        Grid values: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        """
        weight_sets = []
        
        # Define grid values for weights
        grid_values = np.round(np.linspace(0.1, 1.0, 6), 1)  # [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        # Create 6x6 grid (weights x densities ? 36 total)
        for w in grid_values:
            for _ in grid_values:  # loop over densities just to maintain 6x6 length alignment
                set1 = np.full(var_len, w)
                set2 = np.round(1 - set1, 1)
                weight_sets.append((set1, set2))
                if len(weight_sets) >= num_sets:
                    break
            if len(weight_sets) >= num_sets:
                break
        
        # If fewer than num_sets (e.g., <36), tile/repeat
        while len(weight_sets) < num_sets:
            weight_sets.extend(weight_sets[:(num_sets - len(weight_sets))])
        
        return weight_sets
    
    
    @classmethod
    def generate_density_sets(cls, num_sets, var_len):
        """
        Generate density sets for TIES grid search (6x6 = 36 combinations).
        Each set contains identical densities for both models.
        Grid values: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        """
        density_sets = []
        
        # Define grid values for densities
        grid_values = np.round(np.linspace(0.1, 1.0, 6), 1)
        
        # Create full 6x6 grid (for each weight × density pair)
        for _ in grid_values:  # placeholder for weight loop alignment
            for d in grid_values:
                set1 = np.full(var_len, d)
                set2 = np.full(var_len, d)
                density_sets.append((set1, set2))
                if len(density_sets) >= num_sets:
                    break
            if len(density_sets) >= num_sets:
                break
        
        # Ensure list length matches num_sets
        while len(density_sets) < num_sets:
            density_sets.extend(density_sets[:(num_sets - len(density_sets))])
        
        return density_sets

    @classmethod
    def create_directory(cls, base_dir, it_value, yml_dir_name, models_dir_name):
        # Construct the full directory paths
        yml_dir = os.path.join(base_dir,yml_dir_name)
        models_dir = os.path.join(base_dir,models_dir_name)

        # List of directories to create
        dirs = [yml_dir, models_dir]
        
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
                print(f"Directory created: {dir}")
            else:
                print(f"Directory already exists: {dir}")
    @classmethod
    def generate_yml_scripts(cls, weight, density, models, base_model, iteration_name, output_dir='yml_scripts'):
        """
        Generate YAML configuration files for TIES (6x6 grid search = 36 configs).
        Each YAML file merges two models with given weight and density values.
        
        Args:
            weight: list of tuples [(set1, set2), ...] from generate_weight_sets()
            density: list of tuples [(set1, set2), ...] from generate_density_sets()
            models: list of two model names [modelA, modelB]
            base_model: str, which model to treat as base
            iteration_name: int or str, index prefix for filenames
            output_dir: directory to store generated YAMLs
        """
        os.makedirs(output_dir, exist_ok=True)
    
        n = len(weight)
        for i in range(n):
            weights = weight[i]
            densities = density[i]
    
            # Convert NumPy arrays ? lists (for YAML)
            weights_list_0 = np.round(weights[0], 3).tolist()
            weights_list_1 = np.round(weights[1], 3).tolist()
            densities_list_0 = np.round(densities[0], 3).tolist()
            densities_list_1 = np.round(densities[1], 3).tolist()
    
            # Build YAML dictionary (in correct MergeKit format)
            yml_content = {
                'models': [
                    {
                        'model': models[0],
                        'parameters': {
                            'weight': weights_list_0,
                            'density': densities_list_0
                        }
                    },
                    {
                        'model': models[1],
                        'parameters': {
                            'weight': weights_list_1,
                            'density': densities_list_1
                        }
                    }
                ],
                'merge_method': 'ties',
                'base_model': base_model,
                'parameters': {'normalize': True},
                'dtype': 'float16'
            }
    
            # Define filename
            filename = f"individual_{iteration_name}_{i:02d}.yml"
            filepath = os.path.join(output_dir, filename)
    
            # ---- Write YAML file ----
            with open(filepath, 'w') as f:
                f.write("models:\n")
                for m in yml_content['models']:
                    f.write(f"  - model: {m['model']}\n")
                    f.write("    parameters:\n")
                    f.write(f"      weight: {m['parameters']['weight']}   # grid weight\n")
                    f.write(f"      density: {m['parameters']['density']}  # grid density\n")
                f.write(f"merge_method: {yml_content['merge_method']}\n")
                f.write(f"base_model: {yml_content['base_model']}\n")
                f.write("parameters:\n")
                f.write(f"  normalize: {str(yml_content['parameters']['normalize']).lower()}\n")
                f.write(f"dtype: {yml_content['dtype']}\n")
    
        print(f"[INFO] Generated {n} YAML configs in {output_dir}")

    @classmethod
    def run_mergekit(cls, it_value, base_directory, base_output_directory, use_cuda=True, allow_crimes=True):
        # Construct the base directory for YAML files using the it_value
        base_directory_yml = f'{base_directory}'

        # List all files in the base directory
        files = os.listdir(base_directory_yml)
        
        # Filter files matching the pattern individual_it_i.yml
        yml_files = [f for f in files if f.startswith(f'individual_{it_value}_') and f.endswith('.yml')]

        for yml_file in yml_files:
            # Define the full path to the YAML file
            yml_path = os.path.join(base_directory_yml, yml_file)
            
            # Extract the individual_*_** term from the yml_file name
            folder_name = os.path.splitext(yml_file)[0]  # Get the filename without extension
            
            # Define the output directory using the extracted folder name
            output_directory = os.path.join(base_output_directory, folder_name)

            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Construct the command
            command = ['mergekit-yaml', yml_path, output_directory]
            if use_cuda:
                command.append('--cuda')
            if allow_crimes:
                command.append('--allow-crimes')

            # Execute the command
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command)
    @classmethod
    def variation(cls,Parent,lower,upper):
        # Assuming 'Parent' is defined somewhere with dimensions (M, N)
        proC, disC, proM, disM = 1, 20, 1, 20
        M, N = Parent.shape
        # Split the parent array into two parts
        Parent1 = Parent[:M//2, :]
        Parent2 = Parent[M//2:M//2*2, :]
        N,D = Parent1.shape
        
        # Assuming Parent1, Parent2, disC, disM, proC, proM, lower, and upper are defined somewhere
        beta = np.zeros((N, D))
        mu = np.random.rand(N, D)
        
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        
        # Applying random sign
        beta *= np.random.choice([-1, 1], size=(N, D))
        
        # Setting beta to 1 with probability 0.5
        beta[np.random.rand(N, D) < 0.5] = 1
        
        # Applying the crossover probability proC
        beta[np.random.rand(N, D) > proC] = 1
        
        # Generating offspring
        Offspring = np.vstack([
            (Parent1 + Parent2) / 2 + beta * (Parent1 - Parent2) / 2,
            (Parent1 + Parent2) / 2 - beta * (Parent1 - Parent2) / 2
        ])
        
        # Polynomial mutation
        Lower = np.tile(lower, (2 * N, 1))
        Upper = np.tile(upper, (2 * N, 1))
        Site = np.random.rand(2 * N, D) < proM / D
        mu = np.random.rand(2 * N, D)
        
        temp = Site & (mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            (2 * mu[temp] + (1 - 2 * mu[temp]) * 
             (1 - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1)) - 1
        )
        
        temp = Site & (mu > 0.5)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (
            1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * 
                 (1 - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1))
        )
        return Offspring
    @classmethod
    def euclidean_distances(cls,data):
        # Compute all pairs of Euclidean distances
        dist_matrix = np.sqrt(((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2))
        return dist_matrix
    @classmethod
    def mating(cls,combined_array):
        # Calculate distance matrix
        dist_matrix = cls.euclidean_distances(combined_array)
        # Find indices of the 3 closest vectors for each row (excluding the row itself)
        closest_indices = np.argsort(dist_matrix, axis=1)[:, 1:3]  # skip the first column since it is the distance to itself (0)
        # Select a random index and number from each row
        random_selection_with_index = [
            (np.random.randint(len(row)), row[idx])
            for row in closest_indices
            for idx in [np.random.randint(len(row))]
        ]
        
        # Store the result in a variable
        selection_results = []
        for idx, (selected_idx, selected_number) in enumerate(random_selection_with_index):
            selection_results.append((selected_idx, selected_number))
        return selection_results
    @classmethod
    def ens_ss(cls,PopObj, nSort):
        # Remove duplicate rows and get unique rows with their indices
        PopObj, unique_indices = np.unique(PopObj, axis=0, return_inverse=True)
        # Compute the histogram of unique indices
        Table = np.bincount(unique_indices)
        N, M = PopObj.shape
        FrontNo = np.full(N, np.inf)  # Initialize front numbers to infinity
        MaxFNo = 0
        
        while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(unique_indices)):
            MaxFNo += 1
            for i in range(N):
                if np.isinf(FrontNo[i]):
                    Dominated = False
                    for j in range(i):
                        if FrontNo[j] == MaxFNo:
                            m = 1
                            while m < M and PopObj[i, m] >= PopObj[j, m]:
                                m += 1
                            Dominated = m >= M
                            if Dominated or M == 2:
                                break
                    if not Dominated:
                        FrontNo[i] = MaxFNo
        
        # Rearrange FrontNo to match the original order
        FrontNo = FrontNo[unique_indices]
        return FrontNo, MaxFNo
    @classmethod
    def calculate_crowding_distance(cls,PopObj, FrontNo):
        # Get the size of PopObj
        N, M = PopObj.shape
        
        if FrontNo is None:
            FrontNo = np.ones(N, dtype=int)
        
        CrowdDis = np.zeros(N)
        Fronts = np.setdiff1d(np.unique(FrontNo), np.inf)
        
        for f in Fronts:
            Front = np.where(FrontNo == f)[0]
            Fmax = np.max(PopObj[Front, :], axis=0)
            Fmin = np.min(PopObj[Front, :], axis=0)
            
            for i in range(M):
                # Sort indices by the i-th column
                Rank = np.argsort(PopObj[Front, i])
                CrowdDis[Front[Rank[0]]] = np.inf
                CrowdDis[Front[Rank[-1]]] = np.inf
                
                for j in range(1, len(Front) - 1):
                    CrowdDis[Front[Rank[j]]] += (PopObj[Front[Rank[j + 1]], i] - PopObj[Front[Rank[j - 1]], i]) / (Fmax[i] - Fmin[i])
        return CrowdDis
    @classmethod
    def T_ENS(cls,PopObj, nSort):
        # Removing duplicates and keep track of original indices
        unique_PopObj, indices = np.unique(PopObj, axis=0, return_inverse=True)
        Table = np.bincount(indices, minlength=unique_PopObj.shape[0])
        N, M = unique_PopObj.shape
        FrontNo = np.inf * np.ones(N)
        MaxFNo = 0
        Forest = np.zeros(N, dtype=int)
        Children = np.zeros((N, M-1), dtype=int)
        LeftChild = np.ones(N, dtype=int) * M
        Father = np.zeros(N, dtype=int)
        Brother = np.ones(N, dtype=int) * M
        ORank = np.argsort(unique_PopObj[:, 1:M], axis=1)[:, ::-1] + 1  # Reverse order and account for index starting at 1
    
        while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(indices)):
            MaxFNo += 1
            root = np.where(FrontNo == np.inf)[0][0]
            Forest[MaxFNo - 1] = root
            FrontNo[root] = MaxFNo
            for p in range(N):
                if FrontNo[p] == np.inf:
                    Pruning = np.zeros(N, dtype=int)
                    q = Forest[MaxFNo - 1]
                    while True:
                        m = 0
                        while m < M - 1 and unique_PopObj[p, ORank[q, m]] >= unique_PopObj[q, ORank[q, m]]:
                            m += 1
                        if m == M - 1:
                            break
                        else:
                            Pruning[q] = m
                            if LeftChild[q] <= Pruning[q]:
                                q = Children[q, LeftChild[q]]
                            else:
                                while Father[q] and Brother[q] > Pruning[Father[q]]:
                                    q = Father[q]
                                if Father[q]:
                                    q = Children[Father[q], Brother[q]]
                                else:
                                    break
                    if m < M - 1:
                        FrontNo[p] = MaxFNo
                        q = Forest[MaxFNo - 1]
                        while Children[q, Pruning[q]]:
                            q = Children[q, Pruning[q]]
                        Children[q, Pruning[q]] = p
                        Father[p] = q
                        if LeftChild[q] > Pruning[q]:
                            Brother[p] = LeftChild[q]
                            LeftChild[q] = Pruning[q]
                        else:
                            bro = Children[q, LeftChild[q]]
                            while Brother[bro] < Pruning[q]:
                                bro = Children[q, Brother[bro]]
                            Brother[p] = Brother[bro]
                            Brother[bro] = Pruning[q]
    
        FrontNo = FrontNo[indices]
        return FrontNo, MaxFNo
    @classmethod
    # Function to move folders from a source directory to a destination with new naming
    def move_and_rename_folders(cls,source_path, destination_path, start_index=0):
        # List folders in the source directory
        folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
        # Initialize folder index
        i = start_index
        for folder in folders:
            new_folder_name = f"individual_0_0{i}"
            source_folder_path = os.path.join(source_path, folder)
            destination_folder_path = os.path.join(destination_path, new_folder_name)
            # Move and rename the folder
            shutil.move(source_folder_path, destination_folder_path)
            print(f"Moved '{folder}' to '{new_folder_name}'")
            i += 1
        return i  # Return the next starting index for continuity
 
