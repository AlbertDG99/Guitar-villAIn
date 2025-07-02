#!/usr/bin/env python3
"""
Script de Entrenamiento del Agente DQN para Guitar Hero IA
==========================================================
Este script orquesta el entrenamiento del agente de RL.
"""
import time
import numpy as np
from collections import deque
import threading
import keyboard
import os
import logging
import pydirectinput

from src.ai.env import GuitarHeroEnv
from src.ai.dqn_agent import DQNAgent
from src.utils.logger import setup_logger
from src.utils.config_manager import ConfigManager

# --- Variables globales para el logging y la parada segura ---
stop_training_flag = False
latest_info = {}
info_lock = threading.Lock()

def stop_on_q_press(stop_event):
    keyboard.wait('q')
    stop_event.set()
    print("\n'q' pulsada. El entrenamiento se detendrá al final de este episodio.")

def log_status_periodically():
    """Imprime el estado más reciente de la IA cada segundo."""
    global latest_info
    while not stop_training_flag:
        with info_lock:
            if latest_info:
                # \033[K borra el resto de la línea
                print(f"Ep: {latest_info.get('ep', 0)} | "
                      f"Paso: {latest_info.get('step', 0)} | "
                      f"Estado: {latest_info.get('state')} | "
                      f"Recompensa: {latest_info.get('reward', 0):+.2f}\033[K", end='\r')
        time.sleep(1)

def main():
    """Función principal para el entrenamiento del agente."""
    
    print("Iniciando entrenamiento del agente para Guitar Hero IA...")
    
    # --- Configuración ---
    config_manager = ConfigManager()
    ai_config = config_manager.get_ai_config()

    # --- Inicialización del Entorno y Agente ---
    env = GuitarHeroEnv(config_path='config/config.ini')
    
    try:
        # El tamaño del estado es la longitud del vector de observación (teclas + combo)
        if env.observation_space.shape is None:
            raise AttributeError("El shape del espacio de observación es None.")
        state_size = env.observation_space.shape[0]

        # El tamaño de la acción para MultiBinary se obtiene también con shape
        if env.action_space.shape is None:
            raise AttributeError("El shape del espacio de acción es None.")
        action_size = env.action_space.shape[0]

    except AttributeError as e:
        print(f"Error: El espacio de observación o de acción no está bien definido en el entorno. {e}")
        env.close()
        return

    agent = DQNAgent(state_size=state_size, action_size=action_size)

    # --- Bucle de Entrenamiento ---
    start_time = time.time()
    
    for e in range(ai_config['num_episodes']):
        state, _ = env.reset()
        if state is None:
            print("Error: El estado inicial es None. Saltando episodio.")
            continue
            
        state = np.reshape(state, [1, state_size])
        
        total_reward = 0
        
        for time_step in range(ai_config['max_steps_per_episode']):
            action = agent.select_action(state)
            
            next_state, reward, done, _, info = env.step(action)
            
            if next_state is None:
                print(f"Advertencia: next_state es None en el paso {time_step}. Terminando episodio.")
                break

            next_state = np.reshape(next_state, [1, state_size])
            
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        if len(agent.memory) > agent.batch_size:
            agent.train_step()

        # Usar la información final del último paso para el log
        final_score = info.get('score', 0)
        final_combo = info.get('combo', 1)

        print(f"Episodio: {e+1}/{ai_config['num_episodes']}, "
              f"Recompensa: {total_reward:.2f}, "
              f"Score: {final_score}, "
              f"Combo Final: {final_combo}, "
              f"Epsilon: {agent.epsilon:.4f}")

        if (e + 1) % ai_config['save_frequency'] == 0:
            agent.save_model(ai_config['model_save_path'])

    print("Entrenamiento finalizado.")
    env.close()

if __name__ == "__main__":
    try:
        # Un pequeño retraso para que el usuario pueda enfocar la ventana del juego
        print("El entrenamiento comenzará en 5 segundos. Por favor, enfoca la ventana de Guitar Hero.")
        time.sleep(5)
        main()
    except Exception as e:
        main_logger = setup_logger('MainThread')
        main_logger.error(f"Error catastrófico durante el entrenamiento: {e}", exc_info=True)
    finally:
        print("Proceso de entrenamiento finalizado.") 