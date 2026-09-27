# disaSense_project

## Project Objective
This B.Tech minor project aims to perform disaster damage assessment using a Vision-Language Model. By analyzing satellite imagery and related data, the goal is to categorize and evaluate the extent of damage across different regions.

## Dataset
- **Track A:** xBD Dataset
- **Track B:** DisasterM3 Dataset

## Setup Instructions
To set up the environment on Google Colab, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Folder Structure
- `configs/`: Contains configuration files for the project and LoRA settings.
- `data/`: Stores raw datasets (`xbd/`, `disasterm3/`), processed data, and cache.
- `src/`: Source code divided into `preprocessing`, `models`, `training`, `inference`, and `evaluation`.
- `notebooks/`: Jupyter/Colab notebooks for exploratory data analysis and demonstrations.
- `results/`: Output logs, baselines, and confusion matrices.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project overview and setup instructions.

## Experimental Design
We propose a 4-condition experimental design to evaluate the performance of our approach:

| Condition | Description |
|---|---|
| **Condition A** | Zero-shot, no task tokens, no patches |
| **Condition B** | Zero-shot + task tokens, no patches |
| **Condition C** | Zero-shot + task tokens + patches |
| **Condition D** | Fine-tuned + task tokens + patches (proposed method) |
