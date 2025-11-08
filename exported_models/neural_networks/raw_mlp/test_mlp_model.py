#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Lê dataset_nmpc.csv e, para cada episódio, compara a trajetória REAL (x,y)
# com a PREDITA pelo modelo unicycle E pelo modelo MLP treinado.
#
# --- ADAPTADO (v4) ---
# 1. Corrige 'UserWarning' de feature_names do scaler.
# 2. Corrige travamento em 'plt.savefig' ao adicionar 'dropna()'
#    para remover valores NaN do CSV de entrada.

import os
import math
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Use backend não interativo para evitar problemas com Tk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- CONFIG -------------------------
CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_test.csv"

MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_mlp/trained_models/model_epoch_100_batch_64_lr_1e-05_vs_30_hl_64_64_64_64_drop_3"
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")

OUT_DIR  = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/raw_mlp/trained_models/model_epoch_100_batch_64_lr_1e-05_vs_30_hl_64_64_64_64_drop_3/tests"
DT       = 0.02  # s (Usado apenas pelo Unicycle)
os.makedirs(OUT_DIR, exist_ok=True)

REQ_COLS = ["env", "episode", "step", "timestamp",
            "vx", "vy", "wz", "x", "y",
            "qw", "qx", "qy", "qz", "yaw", "v", "w"]

# Configurações do modelo (baseado no parameters.yaml)
MODEL_INPUT_DIM = 5
MODEL_OUTPUT_DIM = 3
MODEL_HIDDEN_LAYERS = [64, 64, 64, 64]


# -------------------- CLASSE DO MODELO MLP --------------------
class MLP(nn.Module):
    """Define a arquitetura da rede MLP."""
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        
        # First hidden layer
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))  # Add dropout after each ReLU
            in_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------- UTILS --------------------------
def wrap_to_pi(angle):
    """Normaliza ângulo para faixa [-pi, pi]."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """
    Propaga o modelo unicycle usando (v,w) passo a passo.
    """
    n = len(v_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x = float(x0)
    y = float(y0)
    th = float(th0)

    x_pred[0] = x
    y_pred[0] = y

    for k in range(n - 1):
        v = float(v_arr[k])
        w = float(w_arr[k])

        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)

        x_pred[k + 1] = x
        y_pred[k + 1] = y

    return x_pred, y_pred


# --- MODIFICADO v4: Adicionado 'input_features' para silenciar o warning ---
def rollout_mlp(x0, y0, th0, v_cmd_arr, w_cmd_arr, 
                model, input_scaler, target_scaler, device, input_features):
    """
    Propaga o estado usando o modelo MLP treinado (modelo de transição de estado).
    """
    n = len(v_cmd_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x = float(x0)
    y = float(y0)
    th = float(th0) 

    x_pred[0] = x
    y_pred[0] = y

    for k in range(n - 1):
        v_cmd = float(v_cmd_arr[k])
        w_cmd = float(w_cmd_arr[k])

        # 1. Preparar input para a MLP
        # --- MODIFICADO v4: Criar DataFrame com nomes de colunas ---
        model_input_df = pd.DataFrame(
            [[x, y, th, v_cmd, w_cmd]], 
            columns=input_features
        )
        
        # 2. Escalar input
        # --- MODIFICADO v4: Usar o DataFrame. Isso silencia o warning ---
        scaled_input = input_scaler.transform(model_input_df)
        
        # 3. Converter para Tensor
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

        # 4. Predizer com a MLP
        with torch.no_grad():
            scaled_output = model(input_tensor)
            
        # 5. Des-escalar output
        pred_output = target_scaler.inverse_transform(scaled_output.cpu().numpy())
        
        x_next, y_next, th_next = pred_output[0]

        # 6. Armazenar posição predita para k+1
        x_pred[k + 1] = x_next
        y_pred[k + 1] = y_next
        
        # 7. Atualizar estado para próximo loop
        x, y, th = x_next, y_next, wrap_to_pi(th_next)

    return x_pred, y_pred


# ------------------------- MAIN ---------------------------
def main():
    # --- Validação dos arquivos ---
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV não encontrado: {CSV_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo MLP não encontrado: {MODEL_PATH}")
    if not os.path.exists(INPUT_SCALER_PATH):
        raise FileNotFoundError(f"Input scaler não encontrado: {INPUT_SCALER_PATH}")
    if not os.path.exists(TARGET_SCALER_PATH):
        raise FileNotFoundError(f"Target scaler não encontrado: {TARGET_SCALER_PATH}")
        
    print(f"[INFO] Carregando CSV: {CSV_PATH}")
    # [cite_start]--- CORREÇÃO: Removido [cite: 1] ---
    df = pd.read_csv(CSV_PATH)

    # Checagem mínima de colunas
    # [cite_start]--- CORREÇÃO: Removido [cite: 1] ---
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
        
    # --- NOVO v4: Remover linhas com NaN para evitar travamento no plot ---
    print(f"[INFO] Formato original do dataset: {df.shape}")
    critical_cols = ['x', 'y', 'yaw', 'v', 'w']
    df.dropna(subset=critical_cols, inplace=True)
    print(f"[INFO] Formato após limpar NaNs: {df.shape}")
    if df.empty:
        print("[ERRO] O DataFrame está vazio após remover NaNs. Abortando.")
        return

    # --- Carregar Modelo e Scalers ---
    print(f"[INFO] Carregando modelo MLP de: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando device: {device}")

    model = MLP(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM, MODEL_HIDDEN_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("[INFO] Carregando scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    # --- NOVO v4: Obter nomes das features para silenciar o warning ---
    input_features = input_scaler.feature_names_in_.tolist()
    
    print(f"   Input features (esperado): {input_features}")
    print(f"   Target features (esperado): {target_scaler.feature_names_in_.tolist()}")

    # --- Processamento dos Episódios ---
    df = df.sort_values(["episode", "env", "step"]).reset_index(drop=True)

    episodes = df["episode"].unique()
    print(f"[INFO] Episódios encontrados: {len(episodes)}")

    for ep in episodes:
        df_ep = df[df["episode"] == ep].copy()

        fig, (ax_traj, ax_error_euclidean, ax_error_accumulated) = plt.subplots(
            3, 1, 
            figsize=(10, 22), 
            gridspec_kw={'height_ratios': [3, 1, 1]}
        )

        # Flag para controlar se algum dado foi plotado
        has_plotted_data = False

        for env_id, df_env in df_ep.groupby("env"):
            df_env = df_env.sort_values("step")
            
            # Checagem de segurança caso o grupo esteja vazio após filtros
            if df_env.empty or len(df_env) < 2:
                print(f"[WARN] Env {env_id} no ep {ep} tem dados insuficientes. Pulando...")
                continue
                
            has_plotted_data = True # Marcamos que temos dados
            
            steps_arr = np.arange(len(df_env))
            x_real = df_env["x"].to_numpy(dtype=float)
            y_real = df_env["y"].to_numpy(dtype=float)
            v_cmd_arr = df_env["v"].to_numpy(dtype=float)
            w_cmd_arr = df_env["w"].to_numpy(dtype=float)
            x0   = float(x_real[0])
            y0   = float(y_real[0])
            yaw0 = float(df_env["yaw"].iloc[0])

            # --- 4. Predição (Unicycle) ---
            x_pred_uni, y_pred_uni = rollout_unicycle(x0, y0, yaw0, v_cmd_arr, w_cmd_arr, DT)
            
            # --- 5. Predição (MLP) ---
            # --- MODIFICADO v4: Passando 'input_features' ---
            x_pred_mlp, y_pred_mlp = rollout_mlp(x0, y0, yaw0, 
                                                 v_cmd_arr, w_cmd_arr, 
                                                 model, input_scaler, target_scaler, 
                                                 device, input_features)

            # --- 6. Calcular Erros ---
            error_uni = np.sqrt(np.square(x_real - x_pred_uni) + np.square(y_real - y_pred_uni))
            error_mlp = np.sqrt(np.square(x_real - x_pred_mlp) + np.square(y_real - y_pred_mlp))
            
            acc_error_uni = np.cumsum(error_uni)
            acc_error_mlp = np.cumsum(error_mlp)


            # --- PLOT 1: Trajetória (ax_traj) ---
            ax_traj.plot(x_real, y_real, label=f"env {env_id} — Real", linewidth=2.5, alpha=0.8)
            ax_traj.plot(x_pred_uni, y_pred_uni, linestyle="--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_pred_mlp, y_pred_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP", linewidth=2)
            
            # --- PLOT 2: Erro Euclidiano (ax_error_euclidean) ---
            ax_error_euclidean.plot(steps_arr, error_uni, linestyle="--", label=f"env {env_id} — Unicycle Err", linewidth=2)
            ax_error_euclidean.plot(steps_arr, error_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP Err", linewidth=2)
            
            # --- PLOT 3: Erro Acumulado (ax_error_accumulated) ---
            ax_error_accumulated.plot(steps_arr, acc_error_uni, linestyle="--", label=f"env {env_id} — Unicycle Acc. Err", linewidth=2)
            ax_error_accumulated.plot(steps_arr, acc_error_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP Acc. Err", linewidth=2)


        # --- Configurações dos Plots (só se tivermos dados) ---
        if has_plotted_data:
            ax_traj.set_aspect("equal")
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.set_title("Comparativo de Trajetória (Real vs. Unicycle vs. MLP)")
            ax_traj.legend(loc="best")
            
            ax_error_euclidean.grid(True, alpha=0.3)
            ax_error_euclidean.set_xlabel("Step (k)")
            ax_error_euclidean.set_ylabel("Erro Euclidiano (m)")
            ax_error_euclidean.set_title("Erro de Predição (Distância Real - Modelo)")
            ax_error_euclidean.legend(loc="best")
            ax_error_euclidean.set_xlim(left=0) # Deixar o matplotlib decidir o limite direito
            ax_error_euclidean.set_ylim(bottom=0)
            
            ax_error_accumulated.grid(True, alpha=0.3)
            ax_error_accumulated.set_xlabel("Step (k)")
            ax_error_accumulated.set_ylabel("Erro Acumulado (m)")
            ax_error_accumulated.set_title("Erro de Predição Acumulado (Soma Cumulativa)")
            ax_error_accumulated.legend(loc="best")
            ax_error_accumulated.set_xlim(left=0) # Deixar o matplotlib decidir o limite direito
            ax_error_accumulated.set_ylim(bottom=0)

            fig.suptitle(f"Episode {int(ep)} — Real vs. Unicycle vs. MLP (dt={DT}s)", fontsize=16)

            out_path = os.path.join(OUT_DIR, f"ep_{int(ep):05d}_compare.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Esta linha não deve mais travar
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[OK] Salvo: {out_path}")
            
        else:
            print(f"[WARN] Nenhum dado válido para plotar no episódio {ep}. Nenhum gráfico salvo.")

        plt.close(fig)

    print("\n[INFO] Processamento concluído.")

if __name__ == "__main__":
    main()