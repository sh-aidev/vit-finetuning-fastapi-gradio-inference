<div align="center">

# VIT Finetuning with FastApi and Gradio Inference

[![python](https://img.shields.io/badge/-Python_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![license](https://img.shields.io/badge/License-MIT-green?logo=mit&logoColor=white)

This repository contains the code for finetuning a Vision Transformer (ViT) model on a custom dataset and deploying it using FastAPI and Gradio for inference.

</div>

## 📌 Feature
- [x] Finetuning VIT model
- [x] Deploying the model using FastApi
- [x] Inference using Gradio

## 📁  Project Structure
The directory structure of new project looks like this:

```
├── LICENSE
├── Makefile
├── README.md
├── __main__.py
├── configs
│   └── config.toml
├── flagged
├── notebooks
│   ├── example.ipynb
│   └── example.py
├── outputs
├── requirements.txt
└── src
    ├── __init__.py
    ├── app.py
    ├── core
    │   ├── __init__.py
    │   ├── inference.py
    │   └── training.py
    ├── server
    │   ├── __init__.py
    │   └── server.py
    └── utils
        ├── __init__.py
        ├── config.py
        ├── logger.py
        ├── models.py
        └── utils.py
```

## 🚀 Getting Started

### Step 1: Clone the repository
```bash
git clone https://github.com/sh-aidev/vit-finetuning-fastapi-gradio-inference.git

cd vit-finetuning-fastapi-gradio-inference
```

### Step 2: Install the required dependencies
```bash
python3 -m pip install -r requirements.txt
```
### Step 3: Run the finetuining script
```bash
# Go to configs and change task_name to "train" to train the model in config.toml

python3 __main__.py

```
### Step 4: Run the Inference
```bash
# Go to configs and change task_name to "infer" to run the inference in config.toml
# Change push_huggingface to true after finetuning is complete to push the model to huggingface

python3 __main__.py

```
### Step 5: Run the FastApi/Gradio server
```bash
# To run the server change task_name to "server" in configs/config.toml
# To run fastapi server change server_type to "fastapi" in configs/config.toml and to run gradion change server_type to "gradio"

python3 __main__.py

```

## 📜  References
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Huggingface Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://www.gradio.app/)
- [PyTorch](https://pytorch.org/)