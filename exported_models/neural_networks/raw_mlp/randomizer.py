#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
randomizer.py
Gera um parameters.yaml com hiperparâmetros aleatórios dentro das faixas especificadas.
"""

import random
import yaml
from typing import List, Dict, Any

_sr = random.SystemRandom()

# ------- Sampling helpers -------

def sample_batch_size() -> int:
    # 16..256 em múltiplos de 16
    return _sr.choice(list(range(16, 257, 16)))

def sample_learning_rate() -> float:
    # 1e-5 até 1e-1 em potências de 10
    return _sr.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

def sample_val_split() -> float:
    # 0.10..0.40 em passos de 0.05
    return _sr.choice([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

def sample_hidden_layers() -> List[int]:
    # De 1 a 6 camadas; cada camada com 32..256 em múltiplos de 32
    n_layers = _sr.randint(1, 6)
    sizes = [_sr.choice(list(range(64, 257, 64))) for _ in range(n_layers)]
    return sizes

# <--- MUDANÇA: Nova função de sampling para dropout
def sample_p_dropout() -> float:
    # 0.0 (desligado) ou valores comuns de dropout
    return _sr.choice([0.0, 0.1, 0.15, 0.2, 0.25])

# ------- Public API -------

def generate_random_config() -> Dict[str, Any]:
    """
    Retorna um dicionário no mesmo esquema do parameters.yaml,
    com campos aleatorizados conforme especificações.
    """
    cfg = {
        "paths": {
            "csv_path": "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc.csv",
            "base_save_dir": "trained_models",
        },
        "training_params": {
            "epochs": 1000,
            "batch_size": 64, # sample_batch_size(),
            "learning_rate": 0.00001, # sample_learning_rate(),
            "val_split": 0.3, # sample_val_split(),
        },
        "model_params": {
            "input_dim": 5,
            "output_dim": 3,
            "hidden_layers": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], # sample_hidden_layers(),
            "p_dropout": 0.3, # sample_p_dropout(),
        },
    }
    return cfg

def generate_and_save_random_yaml(path: str = "parameters.yaml") -> None:
    """
    Gera e salva o YAML no caminho informado.
    """
    cfg = generate_random_config()
    with open(path, "w") as f:
        # Mantém a ordem das chaves e gera YAML simples
        yaml.safe_dump(cfg, f, sort_keys=False)

# Execução direta opcional (útil para testar isoladamente)
if __name__ == "__main__":
    generate_and_save_random_yaml("parameters.yaml")
    print("parameters.yaml aleatório gerado com sucesso.")