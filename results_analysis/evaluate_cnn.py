import torch
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.models.cnn_policy import TraceCNN
from src.envs.LevelForagingEnv import generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv
from src.reasoning.levelbased.l1 import l1_planning

def run_evaluation(model, num_episodes=50, ablate_traces=False, use_heuristic=False):
    """
    Run evaluation episodes.
    If use_heuristic is True, it evaluates the baseline L1 heuristic (ignoring model).
    If ablate_traces is True, channels 3-7 (traces) are zeroed out before passing to the model.
    """
    dim = 10
    nagents = 2
    ntasks = 5
    
    total_rewards = 0
    total_steps = 0
    success_count = 0
    
    for ep in range(num_episodes):
        base_env, _ = generate_random_estimation_scenario(
            method='l1', adhoc_pos=(1, 1), dim=(dim, dim),
            nagents=nagents, ntasks=ntasks, type_knowledge=False,
            parameter_knowledge=True, vision_block=False,
            template_types=['l1', 'l2'], parameters_minmax=[0.5, 1.0],
            seed=None, display=False
        )
        
        # Assign partial observability to the ad hoc agent
        adhoc_agent = base_env.get_adhoc_agent()
        adhoc_agent.radius = 0.25
        adhoc_agent.angle = 1.0
        
        # Match data collection by setting adhoc agent level to 0.9 if needed
        adhoc_agent.level = 0.9

        env = StigmergicLevelForagingEnv(base_env, dim=dim)
        obs_tensor = env.reset()
        
        done = False
        step = 0
        ep_reward = 0
        
        while not done and step < 200:
            if use_heuristic:
                observable_state = base_env.observation_space(base_env.copy())
                action, target = l1_planning(observable_state, base_env.get_adhoc_agent())
            else:
                input_tensor = obs_tensor.unsqueeze(0).clone()
                if ablate_traces:
                    # Zero out trace channels 3, 4, 5, 6, 7
                    input_tensor[:, 3:8, :, :] = 0.0
                    
                with torch.no_grad():
                    pi_logits, _ = model(input_tensor)
                    action = torch.argmax(pi_logits, dim=1).item()
                    
            obs_tensor, reward, done, info = env.step(action)
            ep_reward += reward
            step += 1
            
        total_rewards += ep_reward
        total_steps += step
        if ep_reward > 0:
            success_count += 1
            
    avg_reward = total_rewards / num_episodes
    avg_steps = total_steps / num_episodes
    success_rate = (success_count / num_episodes) * 100
    
    return avg_reward, avg_steps, success_rate

def main():
    model_path = "src/models/trace_cnn_weights.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_cnn.py first.")
        return

    model = TraceCNN(input_channels=9, num_actions=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    num_episodes = 50
    print(f"Starting evaluation ({num_episodes} episodes per setup)...")
    
    print("1. Evaluating Trace CNN (Full Model)...")
    tr_reward, tr_steps, tr_success = run_evaluation(model, num_episodes, ablate_traces=False)
    print(f"   Reward: {tr_reward:.2f} | Steps: {tr_steps:.1f} | Success: {tr_success:.1f}%")
    
    print("2. Evaluating Ablated CNN (No Traces)...")
    ab_reward, ab_steps, ab_success = run_evaluation(model, num_episodes, ablate_traces=True)
    print(f"   Reward: {ab_reward:.2f} | Steps: {ab_steps:.1f} | Success: {ab_success:.1f}%")
    
    print("3. Evaluating Baseline L1 Heuristic...")
    l1_reward, l1_steps, l1_success = run_evaluation(model, num_episodes, use_heuristic=True)
    print(f"   Reward: {l1_reward:.2f} | Steps: {l1_steps:.1f} | Success: {l1_success:.1f}%")
    
    # Plotting
    labels = ['Trace CNN', 'No-Trace CNN', 'L1 Heuristic']
    rewards = [tr_reward, ab_reward, l1_reward]
    steps = [tr_steps, ab_steps, l1_steps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward Plot
    ax1.bar(labels, rewards, color=['#4CAF50', '#F44336', '#2196F3'])
    ax1.set_title('Average Reward per Episode')
    ax1.set_ylabel('Reward (Max 5)')
    ax1.set_ylim(0, 5.5)
    for i, v in enumerate(rewards):
        ax1.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
    # Steps Plot
    ax2.bar(labels, steps, color=['#4CAF50', '#F44336', '#2196F3'])
    ax2.set_title('Average Steps per Episode (Lower is better)')
    ax2.set_ylabel('Steps')
    for i, v in enumerate(steps):
        ax2.text(i, v + 1, f"{v:.1f}", ha='center')
        
    plt.tight_layout()
    plot_path = "results_analysis/Plots/evaluation_plot(adhoc point 09).png"
    plt.savefig(plot_path)
    print(f"\nEvaluation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
