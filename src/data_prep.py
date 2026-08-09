import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # Skip the first row which is just index/X labels, and use the second row as header
    df = pd.read_csv(filepath, header=1)
    df.rename(columns={'default payment next month': 'Y'}, inplace=True)
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates().copy()
    
    # Handle negative values in fields where they shouldn't exist
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    for col in pay_amt_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            
    return df

def prepare_bnn_data(df, target_col='Y'):
    df_nn = df.copy()
    
    # Identify feature categories
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Skewed monetary/limit fields to log-transform
    monetary_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    
    # Log transform monetary fields (handle negative values with sign)
    for col in monetary_cols:
        if col in df_nn.columns:
            df_nn[col] = np.sign(df_nn[col]) * np.log1p(np.abs(df_nn[col]))
            
    # One-hot encode categorical features
    df_nn = pd.get_dummies(df_nn, columns=cat_cols, drop_first=True)
    
    # Ensure boolean columns are integers (0 or 1)
    for col in df_nn.columns:
        if df_nn[col].dtype == bool:
            df_nn[col] = df_nn[col].astype(int)
            
    return df_nn

def prepare_bn_data(df, target_col='Y'):
    df_bn = df.copy()
    
    # Identify continuous columns to bin
    continuous_cols = ['AGE', 'LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    
    # Bin continuous features into 4 quantile buckets
    for col in continuous_cols:
        if col in df_bn.columns:
            df_bn[col] = pd.qcut(df_bn[col], q=4, labels=False, duplicates='drop')
            
    return df_bn

def main():
    raw_path = "data/raw/default_of_credit_card_clients.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data(raw_path)
    print(f"Loaded dataset with shape: {df.shape}")
    
    print("Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned dataset shape: {df.shape}")
    
    # Define target
    target_col = 'Y'
    
    # Split raw data first to ensure exact same split indices for both discrete & continuous representations
    train_idx, test_idx = train_test_split(df.index, test_size=0.30, random_state=42, stratify=df[target_col])
    val_idx, test_idx = train_test_split(test_idx, test_size=0.50, random_state=42, stratify=df.loc[test_idx, target_col])
    
    print(f"Split indices - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create BN discrete version
    print("Preparing discrete data for BN...")
    df_discrete = prepare_bn_data(df, target_col)
    
    train_discrete = df_discrete.loc[train_idx]
    val_discrete = df_discrete.loc[val_idx]
    test_discrete = df_discrete.loc[test_idx]
    
    # Save BN discrete splits
    train_discrete.to_csv(os.path.join(processed_dir, "train_discrete.csv"), index=False)
    val_discrete.to_csv(os.path.join(processed_dir, "val_discrete.csv"), index=False)
    test_discrete.to_csv(os.path.join(processed_dir, "test_discrete.csv"), index=False)
    
    # Create NN continuous version
    print("Preparing continuous data for BNN...")
    df_continuous = prepare_bnn_data(df, target_col)
    
    # For neural network, we standardize continuous features based on the training split only
    # Scaler fit on train only to prevent data leakage
    train_continuous = df_continuous.loc[train_idx].copy()
    val_continuous = df_continuous.loc[val_idx].copy()
    test_continuous = df_continuous.loc[test_idx].copy()
    
    # Identify columns to scale (exclude one-hot and Y)
    exclude_cols = ['Y'] + [col for col in df_continuous.columns if 'SEX_' in col or 'EDUCATION_' in col or 'MARRIAGE_' in col]
    scale_cols = [col for col in df_continuous.columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    train_continuous[scale_cols] = scaler.fit_transform(train_continuous[scale_cols])
    val_continuous[scale_cols] = scaler.transform(val_continuous[scale_cols])
    test_continuous[scale_cols] = scaler.transform(test_continuous[scale_cols])
    
    # Save BNN continuous splits
    train_continuous.to_csv(os.path.join(processed_dir, "train_continuous.csv"), index=False)
    val_continuous.to_csv(os.path.join(processed_dir, "val_continuous.csv"), index=False)
    test_continuous.to_csv(os.path.join(processed_dir, "test_continuous.csv"), index=False)
    
    print("Preprocessing complete. All files saved to data/processed/")

if __name__ == "__main__":
    main()
