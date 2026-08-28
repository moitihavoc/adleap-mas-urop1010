<h1 align="center">Stigmergic Traces for Ad Hoc Teamwork in Level-Based Foraging</h1>

> UROP1010 Project by **Nguyen Minh Hien - V202401696** 
>  Built as an extension of the AdLeap-MAS multi-agent simulation framework<a href="#alves2022adleapmas">[1]</a>.

---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Research Objective](#research-objective)
- [Experimental Environment](#experimental-environment)
- [Reproduce the results](#reproduce-the-results)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Create a Python virtual environment](#2-create-a-python-virtual-environment)
    - [3. Installation](#3-installation)
    - [4. Run the experiments](#4-run-the-experiments)
      - [Generate Training Data (if data not available)](#generate-training-data-if-data-not-available)
      - [Train the CNN Models (if model weights not available)](#train-the-cnn-models-if-model-weights-not-available)
      - [Evaluate Both Models](#evaluate-both-models)
      - [Perform Ablation Study](#perform-ablation-study)
- [Architecture, Ablation Study, Evaluation, Results](#architecture-ablation-study-evaluation-results)
- [My Contribution](#my-contribution)
- [Acknowledgement](#acknowledgement)
- [Reference](#reference)

---
# Overview
<p align="center">
<img src="imgs/Stigmergy LBF.png" alt="Project Thumbnail" style="width: 100%; max-width: 800px;">
</p>
With the premise of applying Stigmergy to Ad Hoc Teamwork, this project investigates whether an ad-hoc agent can learn to interpret **Stigmergic Communication Traces**.

The project combines two concepts:
- Ad Hoc Teamwork<a href="#adhocsurvey">[2]</a>: An agent must cooperate without prior knowledge about its teammates.
- Stigmergy<a href="#stigmergy">[3]</a>: an indirect coordination form in which agents leave persistent traces in the environment to stimulate subsequent work
 
The experimental environment is based on **Level-Based Foraging**, where agents with different capability levels must coordinate to collect task/food items whose required levels may exceed those of individual agents.

The agents have **partial observability** of the environment and does not have access to the history or polity of other agents. For the ad-hoc agent, it must solely rely on its current observations and stigmergic traces left in the environment for its policy.
 
---
# Research Objective
The main research question at this stage is:

> Can an ad-hoc agent learn to interpret the semantics of stigmergic traces to improve cooperation with heuristic teammates under partial observability?

The project specifically investigates whether environmental traces, when implemented correctly, can be learned and prove a degree of usefulness, whose extent is meant to be evaluated for different trace values.

---
# Experimental Environment
The environment used in training and evaluation is a wrapper of Level-Based_Foraging environment with stigmergic traces. It entails the following properties:
- Grid size: 10 x 10
- Agents: 2
- Tasks: 5
- Ad-hoc agent position: (1, 1)
- Ad-hoc agent level: 0.9
- Teammate template policies: L1, L2
- Parameter knowledge: Enabled
- Type knowledge: Disabled
- Vision block: Disabled
- Parameter range: [0.5, 1.0]
- Tracefield (Invisible): Intensity, Age, Agent level, Help signal, Claim signal
  
The figure from AdLeap-MAS<a href="#alves2022adleapmas">[1]</a> illustrates the environment in which the problem is proposed:

<p align="center">
<img src="imgs/level-based-foraging.PNG" alt="Project Thumbnail" style="width: 100%; max-width: 800px;">
</p>

---
# Reproduce the results
### 1. Clone the repository

```bash
git clone https://github.com/moitihavoc/adleap-mas-urop1010.git
cd adleap-mas-urop1010
```

### 2. Create a Python virtual environment 

<summary><strong>Option 1: Conda</strong></summary>

```bash
conda create -n adleap_urop1010 python=3.10
conda activate adleap_urop1010
```

<summary><strong>Option 2: Python venv</strong></summary>

- Linux/MacOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- Windows

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Installation

All dependencies are listed in [Requirements](./requirements.txt).
Install and upgrade (if needed) using `pip`: 

```bash
pip install -r requirements.txt --upgrade
```

### 4. Run the experiments

To determine if the ad-hoc agent can learn to interpret the traces, which includes both environmental clues and explicit semantics, the project evaluate 2 CNN models: 1 trained with properly constructed traces, 1 trained with shuffled-value traces. 

Furthermore, to determine which values are the most important to learning traces, the experiment pipline also includes an ablation study section. The pipeline is as follows:

#### Generate Training Data (if data not available)

To generate a new batch of training data, run:

```bash
python3 src/utils.collect_data.py
```

Training data is generated after running 300 episodes with L1 as the planning policy for the ad-hoc agent. 

Further experiment can be done by modifying the policy, or number of training episodes. The final data is stored at `src/Training_Data/training_data_l1.pt`.

#### Train the CNN Models (if model weights not available)

To train new models, run:

```bash
python3 src/utils/train_cnn.py
python3 src/utils/train_cnn_shuffled.py
```

These commands outputs weights for the respective models, which can be loaded upon evaluation.

#### Evaluate Both Models

Both shuffled and unshuffled models are evaluation on the same set of 100 generated scenerios. The results is display in plots, which are saved at `results_analysis/Plots`.

```bash
python3 results_analysis/evaluate_cnn.py
```

#### Perform Ablation Study

An instance of ablation deprive the model's access to certain traces:
- No Trace: Only basic observation
- Basic Trace: Only Intensity, and Age
- Level Trace: Intensity, Age and Level
- Full Trace: Basic observation and unablated trace

Instances are evaluated with a similar method to prior step:

```bash
python3 results_analysis/ablation_trace.py
```

---
# Architecture, Ablation Study, Evaluation, Results
For more details regarding implementation, evaluation and results of experiments, kindly refer to [UROP1010 - Final Report](./UROP_I___Final_Report.pdf).

---
# My Contribution

This project add functionality to the following components:

<summary><strong>Experiment Environments</strong></summary>

- Environment Wrapper with Stigmergic Traces: `src/envs/StigmergicLevelForagingEnv.py`
- Stigmergic Trace properties and dynamics: `src/communication/traces.py`

<summary><strong>CNN Models</strong></summary>

- Convolutional Neural Network Policy & Model Weights: `src/models`

<summary><strong>Evaluation and Ablation</strong></summary>

- Evaluation between models trained on unshuffled and shuffled traces: `results_analysis/evaluate.cnn.py`
- Evaluation among instances of trace values ablation: `results_analysis/ablation_trace.py`

<summary><strong>Utilities</strong></summary>

- Collect training data: `src/utils/collect_data.py`
- Identify visible grid positions based on agents' FOV: `src/utils/find_visible.py`
- Convert observation dictionary of the base LBF environment to trainable tensors: `src/utils/tensor_convert.py`
- Train CNN model on unshuffled data: `src/utils/train_cnn.py`
- Train CNN model on shuffled data: `src/utils/train_cnn_shuffled.py`

---
# Acknowledgement
This project is based on **AdLeap-MAS: An Open-Source Multi-Agent Simulator for Ad-Hoc Reasoning**<a href="#alves2022adleapmas">[1]</a>, developed by do Carmo Alves, Varma, Elkhatib, and Soriano Marcolino.

AdLeap-MAS provides the underlying multi-agent simulation and ad hoc reasoning framework. This repository extends the framework with the research-specific environment components, stigmergic communication mechanism, neural-network policy, data collection, and experimental evaluation described above.

The original framework is available at:

https://github.com/lsmcolab/adleap-mas

Please cite the original AdLeap-MAS publication when using the underlying framework.

---
# Reference
<a name="alves2022adleapmas">[1]</a> do Carmo Alves, M. A., Varma, A., Elkhatib, Y., & Marcolino, L. S. (2022). *AdLeap-MAS: An Open-source Multi-Agent Simulator for Ad-hoc Reasoning*. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS '22). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 1893–1895.

<a name="adhocsurvey">[2]</a> R. Mirsky, I. Carlucho, A. Rahman, E. Fosong, W.Macke, M. Sridharan,
P. Stone, and S. V. Albrecht, “A Survey of Ad Hoc Teamwork
Research,” Aug. 2022, arXiv:2202.10450 [cs.MA]. [Online]. Available:
http://arxiv.org/abs/2202.10450

<a name="stigmergy">[3]</a> F. Heylighen, “Stigmergy as a universal coordination mech-
anism I: Definition and components,” Cognitive Systems Re-
search, vol. 38, pp. 4–13, Jun. 2016. [Online]. Available:
https://linkinghub.elsevier.com/retrieve/pii/S1389041715000327