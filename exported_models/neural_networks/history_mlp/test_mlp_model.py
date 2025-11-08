#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Lê um CSV de teste e compara a trajetória REAL (x,y)
# com a PREDITA pelo modelo unicycle E pelo modelo MLP treinado.
#
# --- ADAPTADO (v5) ---
# 1. Adaptado para 'history_length > 1' (ex: 3).
# 2. Assume que o modelo foi treinado com HISTÓRICO PASSADO:
#    Input: [sa(k), sa(k-1), sa(k-2)] (15 dim)
#    Output: state(k+1) (3 dim)
# 3. 'rollout_mlp' agora mantém um buffer de histórico.
# 4. Carrega arquitetura (layers, dropout) dinamicamente.

import os
import math
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from collections import deque

# Use backend não interativo para evitar problemas com Tk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- CONFIG -------------------------
# --- MUDANÇA: Parâmetros baseados no seu parameters.yaml
HISTORY_LENGTH = 3
MODEL_HIDDEN_LAYERS = [128, 128, 128, 128] # 
MODEL_P_DROPOUT = 0.0 # 

# --- MUDANÇA: Dimensões dinâmicas
MODEL_STATE_ACTION_DIM = 5 # (x, y, yaw, v, w)
MODEL_INPUT_DIM = MODEL_STATE_ACTION_DIM * HISTORY_LENGTH # 5 * 3 = 15
MODEL_OUTPUT_DIM = 3 # (x_next, y_next, yaw_next)

# --- MUDANÇA: Atualize este path para o modelo que você treinou (com H=3)
# O nome do diretório deve ser algo como:
# ..._hist_3_..._hl_128_128_128_128_drop_3
MODEL_BASE_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_mlp/trained_models/model_hist_3_epoch_100_batch_64_lr_1e-05_vs_30_hl_128_128_128_128"

CSV_PATH = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_nmpc_test.csv"
OUT_DIR  = "/home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/history_mlp/trained_models/model_hist_3_epoch_100_batch_64_lr_1e-05_vs_30_hl_128_128_128_128/tests"


# --- Paths dos artefatos (não mudam)
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "mlp_dynamics.pth")
INPUT_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "input_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_BASE_PATH, "target_scaler.joblib")
DT       = 0.02  # s (Usado apenas pelo Unicycle)
os.makedirs(OUT_DIR, exist_ok=True)

REQ_COLS = ["env", "episode", "step", "timestamp",
            "vx", "vy", "wz", "x", "y",
            "qw", "qx", "qy", "qz", "yaw", "v", "w"]


# -------------------- CLASSE DO MODELO MLP --------------------
# --- MUDANÇA: Classe atualizada para bater com mlp_model.py (recebe p_dropout) ---
class MLP(nn.Module):
    """Define a arquitetura da rede MLP (copiado de mlp_model.py)."""
    
    def __init__(self, input_dim, output_dim, hidden_layers, p_dropout: float = 0.0):
        super(MLP, self).__init__()
        
        layers = []
        
        if not hidden_layers:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            if p_dropout > 0:
                layers.append(nn.Dropout(p=p_dropout))
            
            # Camadas intermediárias
            for i in range(len(hidden_layers) - 1):
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                layers.append(nn.ReLU())
                if p_dropout > 0:
                    layers.append(nn.Dropout(p=p_dropout))
            
            # Saída
            layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ------------------------- UTILS --------------------------
def wrap_to_pi(angle):
    """Normaliza ângulo para faixa [-pi, pi]."""
    return (angle + np.pi) % (2*np.pi) - np.pi


def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """
    Propaga o modelo unicycle usando (v,w) passo a passo. (Sem mudança)
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


# --- MUDANÇA (v5): 'rollout_mlp' reescrito para 'history_length' ---
def rollout_mlp(x0, y0, th0, v_cmd_arr, w_cmd_arr, 
                model, input_scaler, target_scaler, device, 
                history_length, input_features):
    """
    Propaga o estado usando o modelo MLP treinado (com histórico).
    Assume modelo: f( [sa(k), sa(k-1), ..., sa(k-H+1)] ) -> state(k+1)
    """
    n = len(v_cmd_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)
    
    # Estado atual (x, y, yaw) - previsto pelo modelo
    current_state = np.array([x0, y0, th0], dtype=float)
    
    # Buffer de histórico para inputs [x, y, yaw, v, w]
    # Usamos 'deque' para gerenciar a janela deslizante
    history_q = deque(maxlen=history_length)
    
    # --- Preenchimento ("Priming") do Buffer ---
    # Preenche o buffer inicial assumindo que o estado e comando
    # antes de k=0 eram iguais a k=0.
    initial_sa_input = np.array([x0, y0, th0, v_cmd_arr[0], w_cmd_arr[0]], dtype=float)
    for _ in range(history_length):
        history_q.append(initial_sa_input)

    # Ponto inicial (k=0)
    x_pred[0] = x0
    y_pred[0] = y0

    # Loop principal (simulação passo a passo)
    for k in range(n - 1):
        # 1. Obter comando real para este passo
        v_cmd = float(v_cmd_arr[k])
        w_cmd = float(w_cmd_arr[k])

        # 2. Construir o input de estado-ação (SA) para o passo k
        #    Usamos o *estado previsto* do passo anterior
        current_sa_input = np.array([
            current_state[0], 
            current_state[1], 
            current_state[2], 
            v_cmd, 
            w_cmd
        ], dtype=float)

        # 3. Adicionar ao buffer (o mais antigo é removido)
        history_q.append(current_sa_input)
        
        # 4. Criar o vetor de input (15-dim)
        #    O buffer 'history_q' contém [sa(k-H+1), ..., sa(k)]
        #    Precisamos inverter para [sa(k), ..., sa(k-H+1)] se o scaler foi
        #    treinado assim (ex: 'x_h0', 'x_h1', ...).
        #    Assumindo que o scaler espera 'h0' (atual) primeiro:
        
        flat_input_list = []
        # Adiciona na ordem inversa: h0 (mais recente), h1, h2 (mais antigo)
        for sa_vector in reversed(history_q):
            flat_input_list.extend(sa_vector)
            
        flat_input_vector = np.array([flat_input_list], dtype=np.float32) # Shape (1, 15)

        # 5. Criar DataFrame para o scaler (evita warnings)
        model_input_df = pd.DataFrame(
            flat_input_vector, 
            columns=input_features
        )
        
        # 6. Escalar, Predizer, Des-escalar
        scaled_input = input_scaler.transform(model_input_df)
        input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

        with torch.no_grad():
            scaled_output = model(input_tensor)
            
        pred_output = target_scaler.inverse_transform(scaled_output.cpu().numpy())
        
        # 7. Obter próximo estado (previsto)
        #    Lembre-se: o modelo previu state(k+1)
        x_next, y_next, th_next = pred_output[0]

        # 8. Armazenar predição para plot
        x_pred[k + 1] = x_next
        y_pred[k + 1] = y_next
        
        # 9. Atualizar 'current_state' para o próximo loop
        current_state[0] = x_next
        current_state[1] = y_next
        current_state[2] = wrap_to_pi(th_next)

    return x_pred, y_pred


# ------------------------- MAIN ---------------------------
def main():
    # --- Validação dos arquivos ---
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV não encontrado: {CSV_PATH}")
    if not os.path.exists(MODEL_BASE_PATH):
        print(f"!!! ATENÇÃO: Diretório do modelo não encontrado: {MODEL_BASE_PATH}")
        print("!!! Verifique se o nome do diretório está correto (incluindo 'hist_3', 'drop_3', etc.)")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo MLP não encontrado: {MODEL_PATH}")
    if not os.path.exists(INPUT_SCALER_PATH):
        raise FileNotFoundError(f"Input scaler não encontrado: {INPUT_SCALER_PATH}")
    if not os.path.exists(TARGET_SCALER_PATH):
        raise FileNotFoundError(f"Target scaler não encontrado: {TARGET_SCALER_PATH}")
        
    print(f"[INFO] Carregando CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
        
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

    # --- MUDANÇA: Instancia o modelo com os parâmetros corretos
    model = MLP(
        input_dim=MODEL_INPUT_DIM,
        output_dim=MODEL_OUTPUT_DIM,
        hidden_layers=MODEL_HIDDEN_LAYERS,
        p_dropout=MODEL_P_DROPOUT
    )
    
    print(f"[INFO] Arquitetura do modelo: Input({MODEL_INPUT_DIM}) -> {MODEL_HIDDEN_LAYERS} -> Output({MODEL_OUTPUT_DIM}) com p_dropout={MODEL_P_DROPOUT}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Importante para desativar o dropout durante a inferência

    print("[INFO] Carregando scalers...")
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    input_features = input_scaler.feature_names_in_.tolist()
    
    # --- MUDANÇA: Validação da dimensão do input
    if len(input_features) != MODEL_INPUT_DIM:
        raise ValueError(
            f"Erro de dimensão: O scaler espera {len(input_features)} inputs, "
            f"mas o modelo está configurado para {MODEL_INPUT_DIM} (H={HISTORY_LENGTH}). "
            "O scaler foi treinado com o 'history_length' correto?"
        )
    
    print(f"   Input features (esperado {MODEL_INPUT_DIM}): {input_features}")
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
        has_plotted_data = False

        for env_id, df_env in df_ep.groupby("env"):
            df_env = df_env.sort_values("step")
            
            if df_env.empty or len(df_env) < 2:
                print(f"[WARN] Env {env_id} no ep {ep} tem dados insuficientes. Pulando...")
                continue
                
            has_plotted_data = True
            
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
            # --- MUDANÇA: Passando 'HISTORY_LENGTH' e 'input_features' ---
            x_pred_mlp, y_pred_mlp = rollout_mlp(x0, y0, yaw0, 
                                                 v_cmd_arr, w_cmd_arr, 
                                                 model, input_scaler, target_scaler, 
                                                 device, HISTORY_LENGTH, input_features)

            # --- 6. Calcular Erros ---
            error_uni = np.sqrt(np.square(x_real - x_pred_uni) + np.square(y_real - y_pred_uni))
            error_mlp = np.sqrt(np.square(x_real - x_pred_mlp) + np.square(y_real - y_pred_mlp))
            
            acc_error_uni = np.cumsum(error_uni)
            acc_error_mlp = np.cumsum(error_mlp)

            # --- PLOTS (Sem mudança na lógica de plot) ---
            ax_traj.plot(x_real, y_real, label=f"env {env_id} — Real", linewidth=2.5, alpha=0.8)
            ax_traj.plot(x_pred_uni, y_pred_uni, linestyle="--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_pred_mlp, y_pred_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP", linewidth=2)
            
            ax_error_euclidean.plot(steps_arr, error_uni, linestyle="--", label=f"env {env_id} — Unicycle Err", linewidth=2)
            ax_error_euclidean.plot(steps_arr, error_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP Err", linewidth=2)
            
            ax_error_accumulated.plot(steps_arr, acc_error_uni, linestyle="--", label=f"env {env_id} — Unicycle Acc. Err", linewidth=2)
            ax_error_accumulated.plot(steps_arr, acc_error_mlp, linestyle=":", color="red", label=f"env {env_id} — MLP Acc. Err", linewidth=2)

        # --- Configurações dos Plots (Sem mudança) ---
        if has_plotted_data:
            ax_traj.set_aspect("equal")
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.set_title(f"Comparativo de Trajetória (H={HISTORY_LENGTH})")
            ax_traj.legend(loc="best")
            
            ax_error_euclidean.grid(True, alpha=0.3)
            ax_error_euclidean.set_xlabel("Step (k)")
            ax_error_euclidean.set_ylabel("Erro Euclidiano (m)")
            ax_error_euclidean.set_title("Erro de Predição (Distância Real - Modelo)")
            ax_error_euclidean.legend(loc="best")
            ax_error_euclidean.set_xlim(left=0)
            ax_error_euclidean.set_ylim(bottom=0)
            
            ax_error_accumulated.grid(True, alpha=0.3)
            ax_error_accumulated.set_xlabel("Step (k)")
            ax_error_accumulated.set_ylabel("Erro Acumulado (m)")
            ax_error_accumulated.set_title("Erro de Predição Acumulado (Soma Cumulativa)")
            ax_error_accumulated.legend(loc="best")
            ax_error_accumulated.set_xlim(left=0)
            ax_error_accumulated.set_ylim(bottom=0)

            fig.suptitle(f"Episode {int(ep)} — Real vs. Unicycle vs. MLP (H={HISTORY_LENGTH})", fontsize=16)

            out_path = os.path.join(OUT_DIR, f"ep_{int(ep):05d}_compare.png")
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[OK] Salvo: {out_path}")
            
        else:
            print(f"[WARN] Nenhum dado válido para plotar no episódio {ep}. Nenhum gráfico salvo.")

        plt.close(fig)

    print("\n[INFO] Processamento concluído.")

if __name__ == "__main__":
    main()