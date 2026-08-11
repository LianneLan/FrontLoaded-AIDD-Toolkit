import numpy as np

class PessimisticABPPScorer:
    """
    Quantifies off-target risk and epistemic uncertainty over sparse ABPP chemoproteomics data 
    using Deep Ensemble variance to penalize Out-of-Distribution (OOD) molecules in RL loops.
    """
    def __init__(self, ensemble_models: list, lambda_offtarget: float = 1.5, gamma_uncertainty: float = 2.0):
        self.models = ensemble_models  # List of trained proxy models
        self.lam = lambda_offtarget
        self.gamma = gamma_uncertainty

    def evaluate_pessimistic_reward(self, mol_features: np.ndarray, base_front_loaded_reward: float) -> tuple:
        """
        Computes uncertainty-aware pessimistic reward for RL generator.
        """
        # Collect predictions across all ensemble members
        predictions = [model.predict(mol_features) for model in self.models]
        
        mu_offtarget = float(np.mean(predictions))          # Estimated off-target liability
        sigma_epistemic = float(np.std(predictions))        # Epistemic uncertainty (OOD penalty)

        # Pessimistic Reward Calculation
        penalty = (self.lam * mu_offtarget) + (self.gamma * sigma_epistemic)
        final_reward = base_front_loaded_reward - penalty
        
        return max(0.0, final_reward), mu_offtarget, sigma_epistemic
