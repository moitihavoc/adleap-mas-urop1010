import torch
import sys
import os
import random
import matplotlib.pyplot as plt

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.models.cnn_policy import TraceCNN
from src.envs.LevelForagingEnv import generate_random_estimation_scenario
from src.envs.StigmergicLevelForagingEnv import StigmergicLevelForagingEnv

'''
=== ABLATION === 
Evaluate the performance of the model with ablated trace properties:
- No Trace
- Basic Trace: Intensity + Age
- Leveled Trace: Intensity + Age + Agent Level
- Full Trace: Intensity + Age + Agent Level + Help Signal + Claim Signal
'''

def create_evaluation_set(num_scenarios=100):
    """Create one reproducible, shared evaluation set for all conditions."""
    random.seed(2026)
    scenarios = []

    for _ in range(num_scenarios):
        base_env, _ = generate_random_estimation_scenario(
            method='l1', adhoc_pos=(1, 1), dim=(10, 10),
            nagents=2, ntasks=5, type_knowledge=False,
            parameter_knowledge=True, vision_block=False,
            template_types=['l1', 'l2'], parameters_minmax=[0.5, 1.0],
            seed=None, display=False
        )

        base_env.reset() # reset to be able to call env.copy()

        adhoc_agent = base_env.get_adhoc_agent()
        adhoc_agent.radius = 0.25
        adhoc_agent.angle = 1.0
        adhoc_agent.level = 0.9
        scenarios.append(base_env)

    return scenarios


def run_evaluation(model, scenarios, ablate=""):
    """
    Run evaluation episodes.
    If use_heuristic is True, it evaluates the baseline L1 heuristic (ignoring model).
    If ablate_traces is True, channels 3-7 (traces) are zeroed out before passing to the model.
    """
    dim = 10
    rewards = []
    steps_list = []
    
    for base_env in scenarios:
        env = StigmergicLevelForagingEnv(base_env.copy(), dim=dim)
        obs_tensor = env.reset()
        
        done = False
        step = 0
        ep_reward = 0
        
        while not done and step < 200:
            input_tensor = obs_tensor.unsqueeze(0).clone()
            if ablate == "no trace":
                # Zero out trace channels 3, 4, 5, 6, 7
                input_tensor[:, 3:8, :, :] = 0.0
            elif ablate == "basic":
                # Zero out trace channels 5, 6, 7
                input_tensor[:, 5:8, :, :] = 0.0
            elif ablate == "level":
                # Zero out trace channels 6, 7
                input_tensor[:, 6:8, :, :] = 0.0
                    
            with torch.no_grad():
                pi_logits, _ = model(input_tensor)
                action = torch.argmax(pi_logits, dim=1).item()
                    
            obs_tensor, reward, done, info = env.step(action)
            ep_reward += reward
            step += 1
            
        rewards.append(ep_reward)
        steps_list.append(step)

    return rewards, steps_list


def summarize(values):
    """Return the sample mean and 95% normal-approximation confidence interval."""
    values = torch.tensor(values, dtype=torch.float64)
    mean = values.mean().item()
    standard_error = values.std(unbiased=True).item() / (len(values) ** 0.5)
    margin = 1.96 * standard_error
    return mean, margin

def main():
    model_path = "src/models/trace_cnn_weights.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_cnn.py first.")
        return

    model = TraceCNN(input_channels=9, num_actions=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    num_scenarios = 100
    scenarios = create_evaluation_set(num_scenarios)
    print(f"Starting ablation on {num_scenarios} shared scenarios...")
    
    print("1. Evaluating Ablated CNN (No Traces)...")
    ab_reward, ab_steps = run_evaluation(model, scenarios, ablate="no trace")
    
    print("2. Evaluating Ablated CNN with Basic Trace (Intensity + Age)...")
    abb_reward, abb_steps = run_evaluation(model, scenarios, ablate="basic")
    
    print("3. Evaluating Ablated CNN with Leveled Trace (Intensity + Age + Level)...")
    abl_reward, abl_steps = run_evaluation(model, scenarios, ablate="level")

    print("4. Evaluating Trace CNN (Full Model)...")
    tr_reward, tr_steps = run_evaluation(model, scenarios)
    
    # Plotting
    labels = ["No-Trace CNN", "Basic-Trace CNN", "Level-Trace CNN", "Full-Trace CNN"]
    colors = ['#4CAF50', '#F44336', '#2196F3', '#ECF321']
    reward_stats = [summarize(values) for values in [ab_reward, abb_reward, abl_reward, tr_reward]]
    step_stats = [summarize(values) for values in [ab_steps, abb_steps, abl_steps, tr_steps]]
    for label, reward_stat, step_stat in zip(labels, reward_stats, step_stats):
        print(f"{label}: reward={reward_stat[0]:.2f} +/- {reward_stat[1]:.2f} "
              f"(95% CI), steps={step_stat[0]:.1f} +/- {step_stat[1]:.1f} (95% CI)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward Plot
    x = list(range(len(labels)))
    reward_means = [stat[0] for stat in reward_stats]
    reward_errors = [stat[1] for stat in reward_stats]
    ax1.bar(x, reward_means, color=colors, alpha=0.85,
            edgecolor='#333333', linewidth=0.8)
    ax1.errorbar(x, reward_means, yerr=reward_errors, fmt='none',
                 ecolor='#222222', elinewidth=1.5, capsize=5)
    ax1.set_xticks(x, labels, rotation=15)
    ax1.set_title('Mean Reward with 95% CI')
    ax1.set_ylabel('Reward (Max 5)')
    ax1.set_ylim(0, 5.5)
    for i, stat in enumerate(reward_stats):
        ax1.text(i, stat[0] + stat[1] + 0.1, f"{stat[0]:.2f}",
                 ha='center', fontsize=9)
        
    # Steps Plot
        step_means = [stat[0] for stat in step_stats]
        step_errors = [stat[1] for stat in step_stats]
        ax2.bar(x, step_means, color=colors, alpha=0.85,
            edgecolor='#333333', linewidth=0.8)
        ax2.errorbar(x, step_means, yerr=step_errors, fmt='none',
                     ecolor='#222222', elinewidth=1.5, capsize=5)
        ax2.set_xticks(x, labels, rotation=15)
    ax2.set_title('Mean Steps with 95% CI')
    ax2.set_ylabel('Steps')
    for i, stat in enumerate(step_stats):
        ax2.text(i, stat[0] + stat[1] + 1, f"{stat[0]:.1f}", ha='center')
        
    plt.tight_layout()
    plot_path = "results_analysis/Plots/evaluation_plot(ablation 4 - 100 eps).png"
    plt.savefig(plot_path)
    print(f"\nEvaluation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
