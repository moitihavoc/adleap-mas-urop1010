import torch
import sys
import os
import matplotlib.pyplot as plt

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.models.cnn_policy import TraceCNN
from src.utils.evaluation import create_evaluation_set, run_evaluation, summarize

'''
=== ABLATION === 
Evaluate the performance of the model with ablated trace properties:
- No Trace
- Basic Trace: Intensity + Age
- Leveled Trace: Intensity + Age + Agent Level
- Full Trace: Intensity + Age + Agent Level + Help Signal + Claim Signal
'''

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
    no_trace = run_evaluation(model, scenarios, ablate="no trace")
    
    print("2. Evaluating Ablated CNN with Basic Trace (Intensity + Age)...")
    basic_trace = run_evaluation(model, scenarios, ablate="basic")
    
    print("3. Evaluating Ablated CNN with Leveled Trace (Intensity + Age + Level)...")
    level_trace = run_evaluation(model, scenarios, ablate="level")

    print("4. Evaluating Trace CNN (Full Model)...")
    full_trace = run_evaluation(model, scenarios)
    
    # Plotting
    labels = ["No-Trace CNN", "Basic-Trace CNN", "Level-Trace CNN", "Full-Trace CNN"]
    colors = ['#4CAF50', '#F44336', '#2196F3', '#ECF321']
    results = [no_trace, basic_trace, level_trace, full_trace]
    reward_stats = [summarize(result['rewards']) for result in results]
    step_stats = [summarize(result['steps']) for result in results]
    success_stats = [summarize(result['successes']) for result in results]
    for label, reward_stat, step_stat, success_stat in zip(
            labels, reward_stats, step_stats, success_stats):
        print(f"{label}: reward={reward_stat[0]:.2f} +/- {reward_stat[1]:.2f} "
              f"(95% CI), steps={step_stat[0]:.1f} +/- {step_stat[1]:.1f} "
              f"(95% CI), success={success_stat[0] * 100:.1f}% +/- "
              f"{success_stat[1] * 100:.1f}% (95% CI)")
    
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
    plot_path = "results_analysis/Plots/retest_evaluation_plot(ablation 4 - 100 eps).png"
    plt.savefig(plot_path)
    print(f"\nEvaluation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
