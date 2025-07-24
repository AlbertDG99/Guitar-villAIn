"""
Training script for the Guitar Hero AI agent.

This module orchestrates the training process for the reinforcement learning agent,
including environment setup, agent initialization, and training loop management.
"""

import threading
import time
from collections import deque

import keyboard
import numpy as np

stop_training_flag = False
latest_info = None


def stop_on_q_press():
    """Stop training when 'q' is pressed."""
    global stop_training_flag
    keyboard.wait('q')
    stop_training_flag = True
    print("🛑 Training stopped by user (Q pressed)")


def log_status_periodically():
    """Log training status periodically."""
    global latest_info
    while not stop_training_flag:
        time.sleep(5)
        if latest_info:
            print(f"\r📊 Step: {latest_info.get('step', 0)}, "
                  f"State: {latest_info.get('state', 'None')}, "
                  f"Reward: {latest_info.get('reward', 0):.2f}", end='\033[K')


def main():
    """
    Main training function.
    
    Initializes the environment and agent, then runs the training loop
    with periodic logging and safe stopping capabilities.
    """
    global stop_training_flag, latest_info
    
    print("🚀 Starting agent training...")
    
    print("📋 Configuration:")
    print("   - Environment: GuitarHeroEnv")
    print("   - Agent: DQNAgent with Double/Dueling DQN")
    print("   - Training: Prioritized Experience Replay")
    
    try:
        from src.ai.env import GuitarHeroEnv
        from src.ai.dqn_agent import DQNAgent
        
        env = GuitarHeroEnv()
        state_size = 6  # Number of lanes
        action_size = 7  # 6 keys + no-op
        
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        
        print(f"✅ State size: {state_size}")
        print(f"✅ Action size: {action_size}")
        
    except Exception as e:
        print(f"❌ Error initializing environment/agent: {e}")
        return
    
    print("🎮 Training Loop:")
    
    stop_thread = threading.Thread(target=stop_on_q_press, daemon=True)
    log_thread = threading.Thread(target=log_status_periodically, daemon=True)
    stop_thread.start()
    log_thread.start()
    
    episode = 0
    total_reward = 0
    
    try:
        while not stop_training_flag:
            state, info = env.reset()
            if state is None:
                print("⚠️ Warning: Initial state is None")
                continue
                
            episode_reward = 0
            step = 0
            
            while not stop_training_flag and step < 5000:  # Max steps per episode
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                agent.store_experience(state, action, reward, next_state, terminated or truncated)
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                step += 1
                
                latest_info = {
                    'step': step,
                    'state': state,
                    'reward': reward
                }
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            episode += 1
            
            if latest_info:
                print(f"\n🎯 Episode {episode}: "
                      f"Reward: {episode_reward:.2f}, "
                      f"Final Combo: {info.get('combo', 0)}")
            
            if episode % 100 == 0:
                agent.save_model(f"models/checkpoint_episode_{episode}.pth")
        
        print(f"\n✅ Training completed after {episode} episodes")
        print(f"📊 Average reward: {total_reward / episode:.2f}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    finally:
        agent.save_model("models/final_model.pth")
        env.close()


if __name__ == "__main__":
    time.sleep(1)
    
    try:
        main()
    except Exception as e:
        print(f"💥 Catastrophic error: {e}")
    finally:
        print("🏁 Training process completed")
