import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Parameters for the weight distribution
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Parameters for the bias distribution
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        self.kl_divergence = 0.0

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        
        # Initialize log variance parameter to correspond to small initial standard deviation
        self.weight_rho.data.fill_(-3.0)
        self.bias_rho.data.fill_(-3.0)

    def forward(self, x):
        # Reparameterization trick
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        epsilon_w = torch.randn_like(self.weight_mu)
        epsilon_b = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * epsilon_w
        bias = self.bias_mu + bias_sigma * epsilon_b
        
        # Compute KL divergence closed-form
        kl_w = 0.5 * torch.sum(2.0 * math.log(self.prior_sigma) - 2.0 * torch.log(weight_sigma) - 1.0 + (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2))
        kl_b = 0.5 * torch.sum(2.0 * math.log(self.prior_sigma) - 2.0 * torch.log(bias_sigma) - 1.0 + (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2))
        
        self.kl_divergence = kl_w + kl_b
        
        return F.linear(x, weight, bias)

class BNN(nn.Module):
    def __init__(self, input_dim, prior_sigma=0.1):
        super().__init__()
        self.fc1 = BayesLinear(input_dim, 64, prior_sigma)
        self.fc2 = BayesLinear(64, 32, prior_sigma)
        self.fc3 = BayesLinear(32, 1, prior_sigma)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def get_kl(self):
        return self.fc1.kl_divergence + self.fc2.kl_divergence + self.fc3.kl_divergence

class FusedDataset(Dataset):
    def __init__(self, continuous_path, latent_path):
        df_cont = pd.read_csv(continuous_path)
        df_lat = pd.read_csv(latent_path)
        
        # Labels are in continuous file
        self.y = df_cont['Y'].values.astype(np.float32)
        
        # Drop Y from features
        X_cont = df_cont.drop(columns=['Y']).values.astype(np.float32)
        X_lat = df_lat.values.astype(np.float32)
        
        # Stage 2: Concatenation / Fusion
        self.X = np.concatenate([X_cont, X_lat], axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def train_model(model, train_loader, val_loader, epochs=150, lr=0.001, kl_weight=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            
            task_loss = criterion(outputs, y_batch)
            kl_loss = model.get_kl()
            # Loss = negative log likelihood (task loss) + lambda * KL divergence
            loss = task_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                task_loss = criterion(outputs, y_batch)
                kl_loss = model.get_kl()
                loss = task_loss + kl_weight * kl_loss
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    return train_losses, val_losses

def main():
    processed_dir = "data/processed"
    models_dir = "outputs/models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading and fusing train datasets...")
    train_dataset = FusedDataset(
        os.path.join(processed_dir, "train_continuous.csv"),
        os.path.join(processed_dir, "latent_z_train.csv")
    )
    
    print("Loading and fusing validation datasets...")
    val_dataset = FusedDataset(
        os.path.join(processed_dir, "val_continuous.csv"),
        os.path.join(processed_dir, "latent_z_val.csv")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    input_dim = train_dataset.X.shape[1]
    print(f"Fused input dimension: {input_dim}")
    
    # Prior regularization coefficient (KL weight) defaults to 1e-3
    model = BNN(input_dim=input_dim, prior_sigma=0.1)
    
    print("Training Bayesian Neural Network discriminator...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=100, lr=0.001, kl_weight=1e-4)
    
    # Save model weights
    model_save_path = os.path.join(models_dir, "bnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save training curves to outputs directory for plotting later
    curves_df = pd.DataFrame({'epoch': range(1, len(train_losses)+1), 'train_loss': train_losses, 'val_loss': val_losses})
    curves_df.to_csv("outputs/bnn_training_curves.csv", index=False)
    print("Training curves saved to outputs/bnn_training_curves.csv")

if __name__ == "__main__":
    main()
