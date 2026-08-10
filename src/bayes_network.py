import os
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC, ExpertKnowledge
from pgmpy.inference import VariableElimination
from tqdm import tqdm

def get_factor_groups():
    return {
        'demo': {
            'target': 'AGE',
            'features': ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']
        },
        'credit': {
            'target': 'LIMIT_BAL',
            'features': ['LIMIT_BAL']
        },
        'pay_status': {
            'target': 'PAY_0',
            'features': ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        },
        'bill': {
            'target': 'BILL_AMT1',
            'features': [f'BILL_AMT{i}' for i in range(1, 7)]
        },
        'pay_amt': {
            'target': 'PAY_AMT1',
            'features': [f'PAY_AMT{i}' for i in range(1, 7)]
        }
    }

def learn_structure(df):
    print("Learning Bayesian Network structure...")
    
    pay_sequence = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
    fixed_edges = []
    for i in range(len(pay_sequence) - 1):
        fixed_edges.append((pay_sequence[i], pay_sequence[i+1]))
        
    est = HillClimbSearch(df)
    
    Y_outgoing_blacklist = [('Y', col) for col in df.columns if col != 'Y']
    
    expert_knowledge = ExpertKnowledge(required_edges=fixed_edges, forbidden_edges=Y_outgoing_blacklist)
    
    best_dag = est.estimate(
        scoring_method=BIC(df),
        expert_knowledge=expert_knowledge,
        max_iter=100
    )
    
    print(f"Structure learning finished. Number of edges: {len(best_dag.edges())}")
    # Convert DAG to DiscreteBayesianNetwork
    best_model = DiscreteBayesianNetwork(best_dag.edges())
    best_model.add_nodes_from(df.columns)
    return best_model

def estimate_parameters(model, df):
    print("Estimating conditional probability distributions (CPDs)...")
    estimator = BayesianEstimator(model, df)
    cpds = estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10)
    model.add_cpds(*cpds)
    # Check if CPDs are valid and consistent
    assert model.check_model(), "Model has invalid structure/parameters"
    return model

def extract_latent_features(model, df, split_name):
    print(f"Extracting latent posteriors for {split_name} split...")
    inference = VariableElimination(model)
    groups = get_factor_groups()
    
    target_states = {}
    for name, group in groups.items():
        target = group['target']
        cpd = model.get_cpds(target)
        target_states[name] = sorted(cpd.state_names[target])
        
    results = []
    markov_blankets = {name: model.get_markov_blanket(group['target']) for name, group in groups.items()}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_latent = {}
        for name, group in groups.items():
            target = group['target']
            mb = markov_blankets[name]
            
            # Evidence: condition on variables in MB, excluding features of the same factor group and Y
            evidence = {node: int(row[node]) for node in mb if node in row.index and node not in group['features'] and node != 'Y'}
            
            try:
                query_res = inference.query(variables=[target], evidence=evidence, show_progress=False)
                probs = [query_res.values[query_res.name_to_no[target][val]] for val in target_states[name]]
            except Exception as e:
                cpd = model.get_cpds(target)
                probs = list(cpd.values)
            
            for state_idx, p in enumerate(probs):
                sample_latent[f"z_{name}_{state_idx}"] = p
                
        results.append(sample_latent)
        
    res_df = pd.DataFrame(results)
    return res_df

def main():
    processed_dir = "data/processed"
    
    train_df = pd.read_csv(os.path.join(processed_dir, "train_discrete.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val_discrete.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_discrete.csv"))
    
    model = learn_structure(train_df)
    model = estimate_parameters(model, train_df)
    
    z_train = extract_latent_features(model, train_df, "train")
    z_val = extract_latent_features(model, val_df, "val")
    z_test = extract_latent_features(model, test_df, "test")
    
    z_train.to_csv(os.path.join(processed_dir, "latent_z_train.csv"), index=False)
    z_val.to_csv(os.path.join(processed_dir, "latent_z_val.csv"), index=False)
    z_test.to_csv(os.path.join(processed_dir, "latent_z_test.csv"), index=False)
    
    print("Stage 1 completed successfully. Latent features saved.")

if __name__ == "__main__":
    main()
