# [Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks](https://arxiv.org/abs/2502.01074)

Code release for paper *Omni-Mol: Exploring Universal Convergent Space for Omni-Molecular Tasks*


## Overview
![main](assests/main.png)

Building generalist models has recently demonstrated remarkable capabilities in diverse scientific domains. Within the realm of molecular learning, several studies have explored unifying diverse tasks across diverse domains. However, negative conflicts and interference between molecules and knowledge from different domain may have a worse impact in threefold. First, conflicting molecular representations can lead to optimization difficulties for the models. Second, mixing and scaling up training data across diverse tasks is inherently challenging. Third, the computational cost of refined pretraining is prohibitively high. To address these limitations, we present Omni-Mol, a scalable and unified LLM-based framework for direct instruction tuning Omni-Mol builds on three key components to tackles conflicts: (1) a unified encoding mechanism for any task input; (2) an active-learning driven data selection strategy that significantly reduces dataset size; (3) a novel design of the adaptive gradient stabilization module and anchor-and-reconcile MoE framework that ensures stable convergence. Experimentally, Omni-Mol achieves state-of-the-art performance across 15 molecular tasks, demonstrates the presence of scaling laws in the molecular domain, and is supported by extensive ablation studies and analyses validating the effectiveness of its design.

## Release

- [ ] OOD任务选取
- [ ] 去掉AGI-Clip
- [ ] AGI不同rank
- [ ] 8个task的baseline
- [ ] 自动化测试脚本
- [x] 纯text训练开发
- [x] 环境一键配置

[2025/2/8] 🔥 We release our first version of code

## Environment Setup
1. Clone the repository and `cd` to the folder

```bash
git clone https://github.com/WillDreamer/OmniMol.git

cd OmniMol
```
2. (Optional) Environment Settings:

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
or
```bash
bash setup_conda.sh
```

3. Install package of OmniMol through 
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

To follow the default setting, please run the code with:
```bash 
bash scripts/mixtrain_auto_eval.sh
```

Actually, we support multiple kinds of training mode in `model_factory.py':

```bash 
MODEL_STAGE_MAP = {
    "lora": create_lora_model,
    "loramoe": create_lora_moe_model, 
    "sequential": load_moe_lora_model_sequential,
    "puretext": create_puer_text_model 
}

```
1️⃣ "lora" represents the pure lora mode without MoE expansion
2️⃣ "loramoe" represents our design of MoE + PEFT
3️⃣ "sequential" represents the continual pre-training mode instead of our unified SFT
4️⃣ "puretext" represents the abltion of merging Graph modality into text prompt


## Evaluation

We support distributed inference

```bash
bash scripts/dist_eval_all_epoch.sh
```

We also support the auto evaluation after training in 
```bash 
bash scripts/mixtrain_auto_eval.sh
```

Please claim the task for evaluation in `TASK_MAP', and the evaluation mode in `MODEL_LOADER_MAP' with `--model_type' in scripts.


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
