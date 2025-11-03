#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_runner.py
(Substitui trainner_runner.py)

Contém a lógica de treinamento SEQUENCIAL com feedback,
conforme o diagrama do usuário.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def train_model(model, train_episodes, val_episodes, epochs, learning_rate, batch_size, seq_len):
    """
    Executa o loop de treinamento sequencial.
    
    Args:
        model (nn.Module): A instância da MLP.
        train_episodes (list): Lista de episódios de treino.
        val_episodes (list): Lista de episódios de validação.
        epochs (int): Número de épocas.
        learning_rate (float): Taxa de aprendizado.
        batch_size (int): Quantos episódios processar antes de 'optimizer.step()'.
        seq_len (int): O tamanho da sequência de input (ex: 3).
    """
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    # input_features = 5 # (x, y, yaw, v, w) <-- Removido (não usado)
    state_features = 3 # (x, y, yaw)

    print(f"Iniciando treino sequencial... (SeqLen: {seq_len})")

    for epoch in range(epochs):
        
        # --- Fase de Treino ---
        model.train()
        random.shuffle(train_episodes) # Embaralha a ORDEM dos episódios
        
        epoch_train_loss = 0.0
        batch_loss = 0.0 # Perda acumulada para o lote
        
        for i, (ep_inputs, ep_targets) in enumerate(train_episodes):
            
            # ep_inputs = (T, 5) tensor com (x, y, yaw, v, w)
            # ep_targets = (T, 3) tensor com (x, y, yaw)
            
            T = ep_inputs.shape[0] # Comprimento total do episódio
            
            # 'h_states' irá guardar o histórico de inputs para a rede
            # (h1, h2, h3, h4, ... do seu diagrama)
            # --- CORREÇÃO: Usamos .clone() para copiar os dados iniciais ---
            # h_states = torch.zeros_like(ep_inputs) # <- Incorreto
            h_states = ep_inputs.clone() # Copia todos os inputs reais
            
            # Precisamos de 'seq_len' passos iniciais
            if T <= seq_len:
                continue # Pula episódios muito curtos
            
            # (Não precisamos mais preencher os 'h_states' iniciais,
            #  pois o .clone() já fez isso)
            
            episode_total_loss = 0.0
            
            # Loop principal do episódio (step-by-step)
            for k in range(seq_len, T):
                
                # 1. Preparar o input da rede: f(h_{k-3}, h_{k-2}, h_{k-1})
                # (ex: k=3. Pega h_states[0, 1, 2])
                input_seq = h_states[k-seq_len : k]
                
                # Concatena a sequência em um vetor único [15]
                input_vec = input_seq.reshape(1, -1) 
                
                # 2. Fazer a predição (z_k)
                z_k = model(input_vec) # Saída é [1, 3] (estado predito)
                
                # 3. Calcular a perda (loss)
                # Compara a predição z_k com o alvo real y_k
                y_k = ep_targets[k].unsqueeze(0) # Pega o alvo real [1, 3]
                
                loss = criterion(z_k, y_k)
                episode_total_loss += loss
                
                # 4. Criar o próximo 'h' (h_k)
                # h_k = [z_k (predito), v_k (real), w_k (real)]
                
                real_actions = ep_inputs[k, state_features:] # Pega [v_k, w_k]
                
                # --- CORREÇÃO: Adicionado .detach() ---
                # Isso "corta" o histórico do gradiente, impedindo o BPTT.
                # A rede agora treina para prever o próximo passo, e não
                # a sequência inteira. Isso é *muito* mais rápido.
                h_k = torch.cat( (z_k.squeeze(0).detach(), real_actions) , dim=0)
                
                # Armazena h_k para o próximo loop
                h_states[k] = h_k
                
            # Fim do episódio
            if T > seq_len:
                batch_loss += (episode_total_loss / (T - seq_len)) # Perda média do episódio

            # --- Atualização do Lote (Backpropagation) ---
            # Se o lote está cheio (ex: 16 episódios) ou se é o último episódio
            if (i + 1) % batch_size == 0 or (i + 1) == len(train_episodes):
                
                # Assegura que temos algo para processar
                if batch_loss > 0:
                    optimizer.zero_grad()
                    
                    # Média da perda no lote
                    # (Se for o último lote, pode não estar cheio)
                    num_episodes_no_lote = (i + 1) % batch_size
                    if num_episodes_no_lote == 0:
                        num_episodes_no_lote = batch_size
                        
                    batch_loss_avg = batch_loss / num_episodes_no_lote
                    
                    batch_loss_avg.backward() # Calcula gradientes
                    optimizer.step()          # Atualiza pesos
                    
                    epoch_train_loss += batch_loss_avg.item()
                    batch_loss = 0.0 # Zera a perda do lote
        
        # --- Fim da Época de Treino ---
        # (Cálculo da média da perda da época)
        num_lotes = max(1, np.ceil(len(train_episodes) / batch_size))
        train_loss_avg = epoch_train_loss / num_lotes
        if epoch > 0:
            train_losses.append(train_loss_avg)

        # --- Fase de Validação (Lógica original está correta) ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            num_val_episodes = 0
            for ep_inputs, ep_targets in val_episodes:
                T = ep_inputs.shape[0]
                if T <= seq_len:
                    continue
                
                num_val_episodes += 1
                ep_val_loss = 0.0
                for k in range(seq_len, T):
                    # Na validação, usamos SEMPRE o histórico real
                    input_seq = ep_inputs[k-seq_len : k] 
                    input_vec = input_seq.reshape(1, -1)
                    
                    z_k = model(input_vec)
                    y_k = ep_targets[k].unsqueeze(0)
                    
                    loss = criterion(z_k, y_k)
                    ep_val_loss += loss.item()
                    
                val_loss += (ep_val_loss / (T - seq_len))
                
        val_loss_avg = val_loss / max(1, num_val_episodes)
        if epoch > 0:
            val_losses.append(val_loss_avg)

        print(f"Época {epoch+1}/{epochs}, Perda Treino: {train_loss_avg:.6f}, Perda Validação: {val_loss_avg:.6f}")

    return train_losses, val_losses