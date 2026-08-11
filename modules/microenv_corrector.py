import numpy as np

class InProteinReactivityEvaluator:
    """
    Physics-guided surrogate model to adjust gas-phase ML-DFT activation barriers
    (Delta G_gas) into effective in-protein reaction barriers (Delta G_protein)
    based on local active-site microenvironment descriptors.
    """
    def __init__(self, alpha_pka: float = 0.59, beta_field: float = 0.12, std_pka: float = 8.5):
        # alpha: RT ln(10) conversion factor at 298K (~0.59 kcal/mol per pKa unit)
        self.alpha = alpha_pka
        self.beta = beta_field
        self.std_pka = std_pka  # Standard Cys/Lys pKa in bulk water (~8.5)

    def calculate_microenvironment_delta(self, target_pka: float, local_electric_field: float, desolvation_penalty: float) -> float:
        """
        Calculates Delta_Delta_G_microenvironment (kcal/mol).
        """
        pka_shift_term = -self.alpha * (self.std_pka - target_pka)
        field_effect_term = -self.beta * local_electric_field
        
        delta_delta_g = pka_shift_term + field_effect_term + desolvation_penalty
        return delta_delta_g

    def predict_in_protein_barrier(self, delta_g_gas_mldft: float, target_pka: float, 
                                  local_electric_field: float, desolvation_penalty: float) -> float:
        """
        Returns final effective in-protein activation barrier Delta G_protein (kcal/mol).
        """
        ddg = self.calculate_microenvironment_delta(target_pka, local_electric_field, desolvation_penalty)
        delta_g_protein = delta_g_gas_mldft + ddg
        return max(0.1, delta_g_protein)  # Physical lower bound limit
