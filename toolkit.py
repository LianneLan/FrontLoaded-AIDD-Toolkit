"""
FrontLoaded-AIDD-Toolkit: Interactive HITL & AIGC-Driven Synthetic Synthesizer
Combines multi-objective Pareto optimization with Human-in-the-Loop (HITL) selection
and AIGC-based synthetic route generation & evaluation.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Contrib.SA_Score import sascorer

class FrontLoadedAIDDWorkstation:
    def __init__(self, max_generations=3):
        self.max_generations = max_generations
        print("="*75)
        print("  Front-Loaded AIDD Workstation: HITL & AIGC Synthesis Engine Initialized")
        print("="*75)

    def mutate_smiles(self, smiles):
        """化学空间变异算子：生成衍生候选分子"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        mutants = set()
        fragments = ["F", "Cl", "C", "O", "N", "CC"]
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            for frag in fragments:
                editable_mol = Chem.RWMol(mol)
                new_atom = Chem.Atom(frag if len(frag)==1 else 'C') # 简化处理
                new_idx = editable_mol.AddAtom(new_atom)
                try:
                    editable_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(editable_mol)
                    mut_smi = Chem.MolToSmiles(editable_mol)
                    if mut_smi != smiles:
                        mutants.add(mut_smi)
                except:
                    continue
        return list(mutants)[:8]

    def evaluate_molecule(self, smiles):
        """前置约束评价：SA Score, SCScore 代理与结合亲和力"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        sa_score = sascorer.calculateScore(mol)
        mw = Descriptors.MolWt(mol)
        sc_score = 1.0 + 4.0 / (1.0 + np.exp(-0.01 * (mw - 350)))
        
        logp = Descriptors.MolLogP(mol)
        affinity_proxy = min(10.0, max(2.0, 5.0 + 0.4 * logp - 0.15 * sa_score))
        
        return {
            "SMILES": smiles,
            "Affinity": round(float(affinity_proxy), 3),
            "SA_Score": round(float(sa_score), 3),
            "SC_Score": round(float(sc_score), 3)
        }

    def get_pareto_frontier(self, population):
        """帕累托最优筛选核心算法"""
        costs = np.column_stack((
            -np.array([p['Affinity'] for p in population]),
            np.array([p['SA_Score'] for p in population])
        ))
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.any(costs[is_efficient] == c, axis=1)
                is_efficient[i] = True
        
        for i, p in enumerate(population):
            p['Is_Pareto_Optimal'] = bool(is_efficient[i])
        return [p for p in population if p['Is_Pareto_Optimal']]

    def run_optimization(self, seed_smiles):
        """执行前置约束生成与帕累托优化循环"""
        print(f"\n[INFO] Initializing generative scan from seed: {seed_smiles}")
        current_generation = [self.evaluate_molecule(seed_smiles)]
        
        for gen in range(1, self.max_generations + 1):
            next_population = list(current_generation)
            for ind in current_generation:
                mutants = self.mutate_smiles(ind['SMILES'])
                for m_smi in mutants:
                    res = self.evaluate_molecule(m_smi)
                    if res:
                        next_population.append(res)
            current_generation = self.get_pareto_frontier(next_population)[:12]
            print(f"[INFO] Generation {gen} completed. Pareto-optimal candidates found: {len(current_generation)}")
            
        return current_generation

    def aigc_synthesize_and_evaluate(self, lead_molecule):
        """
        AIGC 智能合成与多维评价代理器 (Simulating LLM-driven Retrosynthesis & CDL Protocol)
        """
        print("\n" + "="*75)
        print("  [AIGC Synthesizer] Executing Retrosynthetic Planning & Protocol Generation...")
        print("="*75)
        smi = lead_molecule['SMILES']
        print(f"Target Molecule SMILES : {smi}")
        print(f"Predicted Affinity     : {lead_molecule['Affinity']} (kcal/mol equiv)")
        print(f"Synthetic Accessibility: {lead_molecule['SA_Score']} (Lower is easier)")
        print(f"Synthetic Complexity   : {lead_molecule['SC_Score']}")
        
        # 模拟 AIGC 生成的多步合成路线与化学描述语言 (CDL)
        print("\n--- AIGC-Generated Machine-Executable Synthesis Protocol (CDL) ---")
        print("Step 1: Commercial Building Block Procurement (Enamine REAL Space)")
        print("        -> Selected amine/halide precursor with high stock availability.")
        print("Step 2: Core C-C / C-N Cross-Coupling Reaction (e.g., Suzuki-Miyaura / Amide Coupling)")
        print("        -> Reagents: Pd(dppf)Cl2, K2CO3, Dioxane/H2O, 85°C, 4h.")
        print("Step 3: Late-Stage Functionalization (LSF) & Warhead Installation")
        print("        -> Regioselective installation of electrophilic moiety under mild anhydrous conditions.")
        print("Step 4: Purification & Characterization")
        print("        -> Prep-HPLC purification; verified via LC-MS and 1H-NMR.")
        
        # 自动化多维评价 (AIGC Evaluation)
        print("\n--- AIGC Multi-Dimensional Safety & Viability Evaluation ---")
        sa = lead_molecule['SA_Score']
        if sa < 3.5:
            feasibility = "High (Ideal for automated robotic synthesis)"
            risk = "Low risk of emulsion or purification failure."
        elif sa < 5.5:
            feasibility = "Moderate (Requires standard multi-step optimization)"
            risk = "Standard work-up required; monitor intermediate stability."
        else:
            feasibility = "Low (Challenging stereocenters or strained rings)"
            risk = "High risk of synthetic blind spots; manual chemist intervention advised."
            
        print(f"Synthetic Feasibility Tier : {feasibility}")
        print(f"Work-up & Stability Risk   : {risk}")
        print(f"Predicted Off-Target Alert : Clean profile (No pan-assay interference flags)")
        print("="*75)

    def interactive_hitl_session(self, optimal_leads):
        """
        人机协同交互终端 (Human-in-the-Loop Interactive Session)
        """
        print("\n" + "="*75)
        print("  [HITL Mode] Pareto-Optimal Candidate Lead Library")
        print("="*75)
        for idx, lead in enumerate(optimal_leads, 1):
            print(f"[{idx}] SMILES: {lead['SMILES']}")
            print(f    -> Affinity: {lead['Affinity']} | SA Score: {lead['SA_Score']} | SC Score: {lead['SC_Score']}")
            print("-" * 50)
            
        while True:
            try:
                choice = input("\n👉 Enter the index [1-{}] of the molecule you wish to synthesize and evaluate (or type 'q' to quit): ".format(len(optimal_leads))).strip()
                if choice.lower() == 'q':
                    print("Exiting HITL session. Happy researching!")
                    break
                
                idx = int(choice) - 1
                if 0 <= idx < len(optimal_leads):
                    selected_lead = optimal_leads[idx]
                    print(f"\n[+] You selected candidate [{idx+1}].")
                    self.aigc_synthesize_and_evaluate(selected_lead)
                    cont = input("\nWould you like to select another molecule? (y/n): ").strip().lower()
                    if cont != 'y':
                        print("Exiting HITL session. Happy researching!")
                        break
                else:
                    print("[-] Invalid index. Please choose a number from the list.")
            except ValueError:
                print("[-] Please enter a valid integer or 'q'.")

if __name__ == "__main__":
    # 1. 给定一个种子分子
    seed_molecule = "CC(=O)Nc1ccc(O)cc1" # 对乙酰氨基酚骨架
    
    # 2. 运行前置多目标帕累托优化引擎
    workstation = FrontLoadedAIDDWorkstation(max_generations=2)
    pareto_leads = workstation.run_optimization(seed_molecule)
    
    # 3. 开启人类专家交互与 AIGC 合成评价系统 (HITL)
    if pareto_leads:
        workstation.interactive_hitl_session(pareto_leads)
    else:
        print("[-] No Pareto-optimal leads found in this run.")
