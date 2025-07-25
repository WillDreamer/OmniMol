# [Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks](https://arxiv.org/abs/2502.01074)

Code release for paper *Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks*


## Overview
![main](assests/main.png)

Building generalist models has recently demonstrated remarkable capabilities in diverse scientific domains. Within the realm of molecular learning, several studies have explored unifying diverse tasks across diverse domains. However, negative conflicts and interference between molecules and knowledge from different domain may have a worse impact in threefold. First, conflicting molecular representations can lead to optimization difficulties for the models. Second, mixing and scaling up training data across diverse tasks is inherently challenging. Third, the computational cost of refined pretraining is prohibitively high. To address these limitations, we present Omni-Mol, a scalable and unified LLM-based framework for direct instruction tuning Omni-Mol builds on three key components to tackles conflicts: (1) a unified encoding mechanism for any task input; (2) an active-learning driven data selection strategy that significantly reduces dataset size; (3) a novel design of the adaptive gradient stabilization module and anchor-and-reconcile MoE framework that ensures stable convergence. Experimentally, Omni-Mol achieves state-of-the-art performance across 15 molecular tasks, demonstrates the presence of scaling laws in the molecular domain, and is supported by extensive ablation studies and analyses validating the effectiveness of its design.

## Release
[2025/2/8] 🔥 We release our first version of code

## Environment Setup
1. Clone the repository and `cd` to the folder

```bash
git clone https://github.com/WillDreamer/OmniMol.git

cd OmniMol
```
2. Environment Settings:

```bash
# Download the Miniconda installer script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install to $HOME/miniconda3 in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Activate conda (only in the current shell)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accept the ToS for the main channel:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# (Optional) Add conda to your default shell startup
conda init

# Reload shell config
source ~/.bashrc
```

3. Install package through 
```bash
bash setup_omnimol.sh
```

## Weights
Ongoing

## Dataset
### Task list
- "forward"
- "reagent"
- "retrosynthesis"
-  "homolumo"
- "molcap"
- "solvent"
- "catalyst"
- "yield_BH"
- "yield_SM"
- "dqa"
- "scf"
- "logp"
- "weight"
- "tpsa"
- "complexity"
- "experiment"


## Train
### Stage 1 Training of Projector
```bash
bash scripts/pretrain.sh
```
Please refer to `args.py` for detailed parameter explanation.

### Stage 2 MoE + PEFT
```bash 
bash scripts/moe/llama3.2-1b/mixtrain-moelora.sh
```

## Evaluation
Basic evaluation
```bash
bash scripts/eval.sh
```

If you have multiple GPUs, we support distributed inference
```bash
bash scripts/dist_eval.sh
```

## Code Base
We believe a clean, readable and well-commented code can benefit the community, the design of this code base follows this rule, where we provide detailed annotations and simple/efficient code implementation.

The original code base follows LLaVA(https://github.com/haotian-liu/LLaVA.git) and InstructMol(https://github.com/IDEA-XL/InstructMol.git). But we optimized a lot and achieved better efficiency(less VRAM and faster training speed).

## Citation
```bibtex
@misc{hu2025omnimol,
      title={Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks}, 
      author={Chengxin Hu and Hao Li and Yihe Yuan and Zezheng Song and Haixin Wang},
      year={2025},
      eprint={2502.01074},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01074}, 
}
```
