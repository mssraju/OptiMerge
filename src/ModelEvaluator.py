import os
import numpy as np

class ModelEvaluator:
    def __init__(self, base_path, num_models=8):
        """
        Initialize the ModelEvaluator.
        
        Args:
            base_path (str): Path to the directory containing model folders.
            num_models (int): Number of models to evaluate (default: 8).
        """
        self.base_path = base_path
        self.num_models = num_models
        self.gsm_logme = []
        self.he_logme = []
        self.medqa_logme = []
        self.perplexities = []
        self.sparsities = []
        
    def _normalize_sparsity(self, sparsity_values, current_sparsity):
        """Normalize sparsity to [0, 1], favoring moderate-high values."""
        median = np.median(sparsity_values)
        min_val = np.min(sparsity_values)
        max_val = np.max(sparsity_values)
        
        if current_sparsity < median:
            return 0.5 * (current_sparsity - min_val) / (median - min_val)
        else:
            return 0.5 + 0.5 * (current_sparsity - median) / (max_val - median)
    
    def extract_metrics(self):
        """Extract metrics from all model directories."""
        for i in range(self.num_models):
            model_path = os.path.join(self.base_path, f"individual_0_0{i}")
            
            # Extract GSM LogME
            with open(os.path.join(model_path, "LogME.txt"), 'r') as f:
                line = f.readline().strip()
                self.gsm_logme.append(float(line.split(": ")[1]))
            
            # Extract HE LogME
            with open(os.path.join(model_path, "LogMEHE.txt"), 'r') as f:
                line = f.readline().strip()
                self.he_logme.append(float(line.split(": ")[1]))

            # Extract HE LogME
            with open(os.path.join(model_path, "LogMEMEDQA.txt"), 'r') as f:
                line = f.readline().strip()
                self.medqa_logme.append(float(line.split(": ")[1]))
            
            # Extract Perplexity
            with open(os.path.join(model_path, "Perplexity.txt"), 'r') as f:
                line = f.readline().strip()
                self.perplexities.append(float(line.split(": ")[1]))
            
            # Extract Sparsity
            with open(os.path.join(model_path, "Sparsity.txt"), 'r') as f:
                line = f.readline().strip()
                self.sparsities.append(float(line.split(": ")[1]))
    
    def compute_composite_scores(self):
        """
        Compute normalized composite scores for all models.
        Normalizes GSM LogME and HE LogME separately, then averages them.
        
        Returns:
            tuple: (ranked_indices, composite_scores)
        """
        # Normalize GSM LogME (higher is better)
        norm_gsm = (np.array(self.gsm_logme) - np.min(self.gsm_logme)) / \
                   (np.max(self.gsm_logme) - np.min(self.gsm_logme))
        
        # Normalize HE LogME (higher is better)
        norm_he = (np.array(self.he_logme) - np.min(self.he_logme)) / \
                  (np.max(self.he_logme) - np.min(self.he_logme))
        # Normalize MEDQA LogME (higher is better)
        norm_medqa = (np.array(self.medqa_logme) - np.min(self.medqa_logme)) / \
                  (np.max(self.medqa_logme) - np.min(self.medqa_logme))
        
        # Average normalized LogME values
        norm_logme_avg = (norm_gsm + norm_he + norm_medqa) / 3
        print(norm_logme_avg)
        # Normalize Perplexity (lower is better)
        norm_ppl = 1 - (np.array(self.perplexities) - np.min(self.perplexities)) / \
                   (np.max(self.perplexities) - np.min(self.perplexities))
        
        # Normalize Sparsity
        norm_sparsity = np.array([self._normalize_sparsity(self.sparsities, s) for s in self.sparsities])
        
        # Composite score (equal weights)
        composite_scores = 0.333 * norm_logme_avg + 0.333 * norm_ppl + 0.333 * norm_sparsity
        
        # Rank models (descending order)
        ranked_indices = np.argsort(composite_scores)[::-1]
        return ranked_indices, composite_scores
    def evaluate(self):
        """Run full evaluation pipeline."""
        self.extract_metrics()
        ranked_indices, composite_scores = self.compute_composite_scores()
        return ranked_indices, composite_scores