# FrontLoaded-AIDD-Toolkit

A lightweight, open-source Python toolkit designed to support front-loaded synthetic accessibility scoring, protein microenvironment reactivity calibration, epistemic uncertainty quantification, and Pareto-optimal multi-objective filtering for AI-Driven Drug Discovery (AIDD).

## Key Features
- **Front-Loaded Synthetic Feasibility**: Evaluates SA Score and retrosynthetic solvability.
- **In-Protein Microenvironment Correction**: Adjusts gas-phase activation barriers ($\Delta G^\ddagger$) based on local electrostatic fields and residue $\text{p}K_a$ shifts.
- **Pessimistic Off-Target Scorer**: Deploys Deep Ensemble uncertainty quantification ($\sigma_{\text{epistemic}}$) to penalize out-of-distribution molecules.
- **Interactive CDL Protocol Generator**: Generates machine-executable synthesis protocols for target candidate molecules.

## Installation

```bash
git clone [https://github.com/LianneLan/FrontLoaded-AIDD-Toolkit.git](https://github.com/LianneLan/FrontLoaded-AIDD-Toolkit.git)
cd FrontLoaded-AIDD-Toolkit
pip install rdkit numpy scikit-learn
