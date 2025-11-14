#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Orchestrates the full training pipeline:
- Load YAML config
- Prepare data (windows, scaling, split)
- Build model according to config
- Train + save artifacts (model, scalers, plots, config snapshot)
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
    """Drive the training process from the provided configuration dictionary."""
    # -- Parse sub-configs --
    paths_config = config["paths"]
    train_config = config["training_params"]
    model_config = config["model_params"]

    # -- Construct run directory name (tokenized by hyperparams) --
    lr_token = str(train_config["learning_rate"]).replace("0.", "")
    hl_token = "_".join(str(h) for h in model_config["hidden_layers"])
    vs_token = int(round(train_config["val_split"] * 100))
    hist_token = train_config.get("history_length", 1)
    p_drop_token = model_config.get("p_dropout", 0.0)
    drop_str = f"_drop_{str(p_drop_token).replace('0.', '')}" if p_drop_token > 0 else ""

    run_dir_name = (
        f"model"
        f"_hist_{hist_token}"
        f"_epoch_{train_config['epochs']}"
        f"_batch_{train_config['batch_size']}"
        f"_lr_{lr_token}"
        f"_vs_{vs_token}"
        f"_hl_{hl_token}"
        f"{drop_str}"
    )

    save_dir = os.path.join(paths_config["base_save_dir"], run_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "mlp_dynamics.pth")

    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]")
    print(f"Artifacts will be saved in: {save_dir}")

    # -- Snapshot the full config for reproducibility --
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # --------------------- 1) Data preparation ---------------------
    print("\n--- 1. Preparing Data ---")
    history_length = train_config.get("history_length", 1)
    train_data, val_data, _, _, input_dim, output_dim = prepare_data(
        csv_path=paths_config["csv_path"],
        scalers_dir=save_dir,                      # persist scalers alongside model
        val_split_ratio=train_config["val_split"],
        normalize_data=True,
        history_length=history_length,
    )
    if train_data is None:
        print("Data preparation failed. Aborting.")
        return
    print("Data ready.")

    # ---------------------- Device selection -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------- 2) Model construction -------------------
    print("\n--- 2. Initializing Model ---")
    model = MLPDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_config["hidden_layers"],
        p_dropout=model_config.get("p_dropout", 0.0),
    ).to(device)
    print(
        f"Model: Input({input_dim}) -> {model_config['hidden_layers']} "
        f"-> Output({output_dim})  (dropout={model_config.get('p_dropout', 0.0)})"
    )
    print(model)

    # ---------------------- 3) Train loop ---------------------------
    print("\n--- 3. Starting Training ---")
    train_losses, val_losses, train_l1_per_comp, val_l1_per_comp = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=train_config["epochs"],
        learning_rate=train_config["learning_rate"],
        batch_size=train_config["batch_size"],
        device=device,
    )
    print("Training complete.")

    # ---------------------- 4) Save artifacts -----------------------
    print("\n--- 4. Saving Artifacts ---")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_losses(train_losses, val_losses, save_dir=save_dir)
    plot_component_l1(train_l1_per_comp, val_l1_per_comp, save_dir=save_dir)
    print("\nProcess finished successfully.")


if __name__ == "__main__":
    # -- Entry point: load YAML and call main --
    CONFIG_PATH = "parameters.yaml"
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"Configuration loaded from {CONFIG_PATH}")
        main(cfg)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração não encontrado em {CONFIG_PATH}")
    except yaml.YAMLError as e:
        print(f"ERRO: Falha ao parsear o arquivo YAML: {e}")
    except Exception as e:
        print(f"ERRO inesperado: {e}")
