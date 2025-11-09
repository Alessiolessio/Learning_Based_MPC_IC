#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Orchestrates the training:
- Load YAML config
- Prepare data (scalers saved into the run dir)
- Build the MLP
- Train, then save model + plots + config snapshot
"""

import os
import yaml
import torch
from datetime import datetime

from mlp_model import MLPDynamicsModel
from data_preparation import prepare_data
from training_runner import train_model
from plotting import plot_losses, plot_component_l1


def main(config: dict):
    """Run the full training pipeline based on the provided config dict."""
    # -- Resolve device and basic GPU info for logging --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    # -- Unpack sub-configs --
    paths_cfg = config["paths"]
    train_cfg = config["training_params"]
    model_cfg = config["model_params"]

    # -- Build a run folder name based on key hyperparameters --
    lr_token = str(train_cfg["learning_rate"]).replace("0.", "")
    hl_token = "_".join(str(h) for h in model_cfg["hidden_layers"])
    vs_token = int(round(train_cfg["val_split"] * 100))
    p_drop = model_cfg.get("p_dropout", 0.0)
    drop_str = f"_drop_{str(p_drop).replace('0.', '')}" if p_drop > 0 else ""

    run_dir_name = (
        f"model"
        f"_epoch_{train_cfg['epochs']}"
        f"_batch_{train_cfg['batch_size']}"
        f"_lr_{lr_token}"
        f"_vs_{vs_token}"
        f"_hl_{hl_token}"
        f"{drop_str}"
    )
    save_dir = os.path.join(paths_cfg["base_save_dir"], run_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "mlp_dynamics.pth")

    print("\n" + "=" * 80)
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting run")
    print("=" * 80)
    print(f"All artifacts (model, scalers, plots) will be saved in: {save_dir}")

    # -- Persist a config snapshot for reproducibility --
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # -------------------- 1) Data preparation --------------------
    print("\n--- 1. Preparing Data ---")
    train_data, val_data, _, _ = prepare_data(
        csv_path=paths_cfg["csv_path"],
        scalers_dir=save_dir,
        val_split_ratio=train_cfg["val_split"],
        normalize_data=True,
    )
    if train_data is None:
        print("Data preparation failed. Aborting.")
        return
    print("Data ready.")

    # -------------------- 2) Model initialization ----------------
    print("\n--- 2. Initializing Model ---")
    model = MLPDynamicsModel(
        input_dim=model_cfg["input_dim"],
        output_dim=model_cfg["output_dim"],
        hidden_layers=model_cfg["hidden_layers"],
        p_dropout=p_drop,
    )
    print(
        f"Model: Input({model_cfg['input_dim']}) -> {model_cfg['hidden_layers']} "
        f"-> Output({model_cfg['output_dim']})  (dropout={p_drop})"
    )
    print(model)

    # -------------------- 3) Training loop ----------------------
    print("\n--- 3. Starting Training ---")
    train_losses, val_losses, train_l1_per_comp, val_l1_per_comp = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
        device=device,
    )
    print("Training complete.")

    # -------------------- 4) Save artifacts ---------------------
    print("\n--- 4. Saving Artifacts ---")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_losses(train_losses, val_losses, save_dir=save_dir)
    plot_component_l1(train_l1_per_comp, val_l1_per_comp, save_dir=save_dir)

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    # -- Entrypoint: read YAML and call 'main' --
    config_path = "parameters.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        main(config)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração não encontrado em {config_path}")
    except yaml.YAMLError as e:
        print(f"ERRO: Falha ao parsear o arquivo YAML: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a execução: {e}")
