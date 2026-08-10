import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE
import seaborn as sns

from bnn_model import BNN, FusedDataset, train_model

def get_fused_loader(continuous_path, latent_path, batch_size=256, shuffle=False):
    dataset = FusedDataset(continuous_path, latent_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset

def run_mc_inference(model, loader, T=50):
    model.eval()
    all_predictions = []
    y_true = []
    
    # We will accumulate predictions for each forward pass
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_true.extend(y_batch.numpy())
            
            # Draw T forward passes
            batch_preds = []
            for _ in range(T):
                preds = model(X_batch).squeeze().numpy()
                batch_preds.append(preds)
            
            # Shape: (T, batch_size)
            all_predictions.append(np.array(batch_preds))
            
    # Concatenate along batch dimension
    # Shape: (T, num_samples)
    y_preds_mc = np.hstack(all_predictions)
    y_true = np.array(y_true)
    
    # Calculate mean and variance per sample
    risk_scores = np.mean(y_preds_mc, axis=0)
    uncertainty = np.var(y_preds_mc, axis=0)
    
    return y_true, risk_scores, uncertainty

def plot_loss_curves():
    print("Plotting loss curves...")
    curves_df = pd.read_csv("outputs/bnn_training_curves.csv")
    
    plt.figure(figsize=(8, 5))
    plt.plot(curves_df['epoch'], curves_df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(curves_df['epoch'], curves_df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    plt.title('Loss vs. Epoch during BNN Training', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/loss_vs_epoch.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {save_path}")

def plot_tsne(dataset, uncertainty, y_true):
    print("Performing t-SNE projection...")
    X_fused = dataset.X
    
    # Subsample if test set is too large for fast t-SNE
    max_samples = 2000
    if len(X_fused) > max_samples:
        indices = np.random.choice(len(X_fused), max_samples, replace=False)
        X_sub = X_fused[indices]
        y_sub = y_true[indices]
        u_sub = uncertainty[indices]
    else:
        X_sub = X_fused
        y_sub = y_true
        u_sub = uncertainty
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_sub)
    
    plt.figure(figsize=(10, 8))
    
    # Define high-uncertainty threshold (75th percentile)
    threshold = np.percentile(u_sub, 75)
    high_u = u_sub > threshold
    
    # Plot normal points
    sns.scatterplot(
        x=X_2d[~high_u, 0], y=X_2d[~high_u, 1],
        hue=y_sub[~high_u], palette={0: '#1f77b4', 1: '#ff7f0e'},
        alpha=0.6, style=y_sub[~high_u], markers={0: 'o', 1: 's'}, s=40
    )
    
    # Plot high uncertainty points as larger, translucent red 'x' marks
    plt.scatter(
        X_2d[high_u, 0], X_2d[high_u, 1],
        color='#d62728', marker='x', s=60, alpha=0.8,
        label='High Uncertainty (Outliers)'
    )
    
    plt.title('t-SNE Visualization of Fused Representation Space', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(title='Category', fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = "outputs/figures/tsne_visualization.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")

def run_lambda_sweep(input_dim, train_loader, val_loader):
    print("Running KL regularization coefficient (lambda) sweep...")
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0]
    aurocs = []
    
    for lmb in lambdas:
        print(f"Sweeping lambda = {lmb}...")
        # Train for 20 epochs per lambda to keep the sweep fast
        model = BNN(input_dim=input_dim, prior_sigma=0.1)
        train_model(model, train_loader, val_loader, epochs=25, lr=0.002, kl_weight=lmb)
        
        # Evaluate on validation loader
        y_val, risk_val, _ = run_mc_inference(model, val_loader, T=30)
        auroc = roc_auc_score(y_val, risk_val)
        aurocs.append(auroc)
        print(f"Lambda {lmb} -> Val AUROC: {auroc:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, aurocs, marker='o', color='#2ca02c', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.title('Sensitivity of Prior Regularization Coefficient (λ) to AUROC', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Prior Regularization Coefficient λ (Log Scale)', fontsize=12)
    plt.ylabel('Validation AUROC', fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    save_path = "outputs/figures/prior_regularization_sensitivity.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Lambda sweep curve saved to {save_path}")

def main():
    processed_dir = "data/processed"
    models_dir = "outputs/models"
    
    print("Loading test datasets...")
    test_loader, test_dataset = get_fused_loader(
        os.path.join(processed_dir, "test_continuous.csv"),
        os.path.join(processed_dir, "latent_z_test.csv")
    )
    
    input_dim = test_dataset.X.shape[1]
    model = BNN(input_dim=input_dim, prior_sigma=0.1)
    
    model_path = os.path.join(models_dir, "bnn_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Train BNN first.")
        return
        
    model.load_state_dict(torch.load(model_path))
    
    print("Running MC sampling inference (T=50 passes)...")
    y_true, risk_scores, uncertainty = run_mc_inference(model, test_loader, T=50)
    
    # Classify based on 0.5 threshold
    y_pred = (risk_scores >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, risk_scores)
    
    print("\n================ TEST SET METRICS ================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUROC:     {auroc:.4f}")
    print("==================================================\n")
    
    # Save metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'F1-Score', 'AUROC'],
        'Value': [accuracy, precision, f1, auroc]
    })
    metrics_df.to_csv("outputs/evaluation_metrics.csv", index=False)
    
    # Plot loss curves
    plot_loss_curves()
    
    # Plot t-SNE
    plot_tsne(test_dataset, uncertainty, y_true)
    
    # Load train and val for lambda sweep
    train_loader, _ = get_fused_loader(
        os.path.join(processed_dir, "train_continuous.csv"),
        os.path.join(processed_dir, "latent_z_train.csv"),
        batch_size=128, shuffle=True
    )
    val_loader, _ = get_fused_loader(
        os.path.join(processed_dir, "val_continuous.csv"),
        os.path.join(processed_dir, "latent_z_val.csv"),
        batch_size=256, shuffle=False
    )
    
    # Run lambda sweep
    run_lambda_sweep(input_dim, train_loader, val_loader)

if __name__ == "__main__":
    main()
