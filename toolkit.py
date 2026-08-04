"""
FrontLoaded-AIDD-Toolkit: A Lightweight Python Pipeline for Front-Loaded
Synthetic Accessibility Scoring and Pareto-Optimal Multi-Objective Filtering.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer  # RDKit内置的 SA Score 评估模块

class FrontLoadedAIDDPipeline:
    def __init__(self, affinity_weights=(0.5, 0.5)):
        """
        初始化前置多目标优化过滤器
        """
        self.weights = affinity_weights

    def calculate_sa_score(self, smiles_list):
        """
        计算输入的分子列表的 SA Score (Synthetic Accessibility Score)
        分数范围: 1 (极易合成) 至 10 (极难合成)
        """
        sa_scores = []
        valid_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                score = sascorer.calculateScore(mol)
                sa_scores.append(score)
                valid_smiles.append(smi)
            else:
                sa_scores.append(None)
        return valid_smiles, sa_scores

    def mock_sc_score(self, smiles_list):
        """
        SCScore 包装器模拟器（实际应用中可接入官方开源的神经网络模型预测值）
        值域: 1.0 (简单) 至 5.0 (复杂)
        """
        # 此处以分子量和原子数进行符合化学直觉的复杂性代理模拟，方便用户零配置跑通
        sc_scores = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                # 简单的启发式代理公式，模拟 USPTO 训练的反应步数复杂度
                score = 1.0 + 4.0 / (1.0 + np.exp(-0.01 * (mw - 300)))
                sc_scores.append(round(score, 3))
            else:
                sc_scores.append(None)
        return sc_scores

    def extract_pareto_frontier(self, affinity_scores, complexity_scores):
        """
        多目标优化：提取帕累托前沿（Pareto Frontier）
        目标 1: 最大化靶点结合亲和力 (Affinity) -> 转换为最小化负亲和力
        目标 2: 最小化合成复杂度 (Complexity/SA或SCScore)
        """
        costs = np.column_stack((-np.array(affinity_scores), np.array(complexity_scores)))
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # 支配判断：如果存在其他点在所有目标上都优于或等于当前点，则当前点不是 Pareto 优化的
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.any(costs[is_efficient] == c, axis=1)
                is_efficient[i] = True
        return is_efficient

    def run_screening_pipeline(self, smiles_list, mock_affinities):
        """
        运行完整的端到端前置合成可行性筛选流水线
        """
        print("[INFO] Running RDKit SA Score Evaluation...")
        valid_smi, sa_scores = self.calculate_sa_score(smiles_list)
        
        print("[INFO] Computing Synthetic Complexity (SCScore Proxy)...")
        sc_scores = self.mock_sc_score(valid_smi)
        
        print("[INFO] Executing Multi-Objective Pareto-Frontier Filtering...")
        pareto_mask = self.extract_pareto_frontier(mock_affinities, sc_scores)
        
        results = []
        for i, smi in enumerate(valid_smi):
            results.append({
                "SMILES": smi,
                "Affinity": mock_affinities[i],
                "SA_Score": sa_scores[i],
                "SC_Score": sc_scores[i],
                "Is_Pareto_Optimal": bool(pareto_mask[i])
            })
        return results

# ==========================================
# 示例运行脚本
# ==========================================
if __name__ == "__main__":
    # 测试用的 AI 生成分子 SMILES 列表
    test_smiles = [
        "CC(=O)Nc1ccc(O)cc1",             # 对乙酰氨基酚 (简单)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # 布洛芬 (中等)
        "CN1C2CC(C3=CC=CC=C3C2)C4=CC=CC=C41", # 较复杂的生物碱类虚拟分子
        "C1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C4=CC=CC=C4" # 复杂多环虚拟分子
    ]
    # 模拟 AI 模型预测的结合亲和力分数 (假设值 2.0 ~ 10.0)
    mock_affinities = [4.5, 6.2, 8.1, 9.0]

    pipeline = FrontLoadedAIDDPipeline()
    output_results = pipeline.run_screening_pipeline(test_smiles, mock_affinities)

    print("\n--- Pipeline Execution Results ---")
    for res in output_results:
        print(res)
