import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.getcwd())

from src.models.cnn_policy import TraceCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    data_path = "src/Training_Data/training_data_l1.pt"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print("Loading data...")
    episodes = torch.load(data_path, weights_only=False)
    
    # Flatten episodes into a list of transitions
    transitions = []
    for ep in episodes:
        transitions.extend(ep)
        
    print(f"Total transitions loaded: {len(transitions)}")
    
    # Prepare tensors
    # obs is already a tensor (1, 9, 10, 10). We cat them along dim 0.
    obs_batch = torch.stack([t['obs'] for t in transitions], dim=0).to(device)
    
    # target actions for cross entropy loss
    action_batch = torch.tensor([t['action'] for t in transitions], dtype=torch.long).to(device)
    
    # target values for MSE loss
    value_batch = torch.tensor([t['value'] for t in transitions], dtype=torch.float32).unsqueeze(1).to(device)
    
    # Initialize Model, Loss, Optimizer
    model = TraceCNN(input_channels=9, num_actions=5).to(device)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 128
    num_epochs = 30
    dataset_size = obs_batch.size(0)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Shuffle indices
        indices = torch.randperm(dataset_size)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        
        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            b_obs = obs_batch[batch_idx]
            b_actions = action_batch[batch_idx]
            b_values = value_batch[batch_idx]
            
            # Forward pass
            pi_logits, v_pred = model(b_obs)
            
            # Compute losses
            loss_pi = policy_criterion(pi_logits, b_actions)
            loss_v = value_criterion(v_pred, b_values)
            
            # Combine loss (weight value loss by 0.5)
            loss = loss_pi + 0.5 * loss_v
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_policy_loss += loss_pi.item()
            epoch_value_loss += loss_v.item()
            epoch_total_loss += loss.item()
            
        # Logging
        num_batches = (dataset_size + batch_size - 1) // batch_size
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Total Loss: {epoch_total_loss/num_batches:.4f} "
              f"(Policy: {epoch_policy_loss/num_batches:.4f}, Value: {epoch_value_loss/num_batches:.4f})")
        
    # Save the model
    os.makedirs("src/models", exist_ok=True)
    save_path = "src/models/trace_cnn_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
