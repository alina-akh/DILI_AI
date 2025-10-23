import pandas as pd
import datamol as dm
import numpy as np
import example_submission as tox

def compute_scaffolds(df):
    """Compute generic scaffolds for SMILES in the DataFrame."""
    df["scaffold"] = df["smiles"].apply(tox._generic_scaffold_smiles)
    clusters = df.groupby("scaffold")
    print(f"Total scaffolds found: {len(clusters)}")
    return df

def cluster_molecules(df, cutoff=0.7):
    """Cluster molecules based on similarity."""
    df["mol"] = df["smiles"].map(dm.to_mol)
    df = df[df["mol"].notnull()].reset_index(drop=True)  # Filter out invalid SMILES
    
    clusters, mol_clusters = dm.cluster_mols(df["mol"].tolist(), cutoff=cutoff)
    
    # Map molecule ID to cluster ID
    mol_to_cluster = {}
    for cid, cluster in enumerate(mol_clusters):
        for mol in cluster:
            mol_to_cluster[id(mol)] = cid
    
    df["cluster_id"] = df["mol"].apply(lambda m: mol_to_cluster.get(id(m)))
    return df, mol_clusters


def check_scaffold_integrity(df):
    """Check if scaffold groups are split across multiple clusters."""
    violations = []
    for scaf, group in df.groupby("scaffold"):
        cluster_ids = group["cluster_id"].dropna().unique()
        if len(cluster_ids) > 1:
            violations.append({
                "scaffold": scaf,
                "n_molecules": len(group),
                "clusters": cluster_ids.tolist()
            })
    
    print(f"Total scaffold groups: {df['scaffold'].nunique()}")
    print(f"Scaffold groups split across multiple clusters: {len(violations)}")
    
    if violations:
        print("\nExamples of conflicts:")
        for v in violations[:10]:
            print(f"  Scaffold: {v['scaffold']}")
            print(f"    Molecules: {v['n_molecules']}")
            print(f"    Clusters: {v['clusters']}")
    else:
        print("All scaffold groups fully align with similarity clusters.")
    
    return violations

def merge_clusters(df, n_target_clusters=20, seed=42):
    """Merge existing clusters into a smaller number of target clusters."""
    unique_clusters = df["cluster_id"].unique()
    np.random.seed(seed)
    shuffled_clusters = np.random.permutation(unique_clusters)
    
    new_cluster_mapping = {}
    for i, old_cluster in enumerate(shuffled_clusters):
        new_cluster_mapping[old_cluster] = i % n_target_clusters
    
    df["merged_cluster"] = df["cluster_id"].map(new_cluster_mapping)
    print(df["merged_cluster"].value_counts().sort_index())
    return df


df = pd.read_csv("df_unclustered.csv")
smiles_list = df["smiles"].tolist()

df = compute_scaffolds(df)

if "scaffold" not in df.columns:
    raise ValueError("The DataFrame does not contain the 'scaffold' column!")

df, mol_clusters = cluster_molecules(df)

violations = check_scaffold_integrity(df)

# Print number of clusters
n_clusters = len(mol_clusters)
print(f"Number of similarity clusters: {n_clusters}")

n_clusters_df = df["cluster_id"].nunique()
print(f"Number of unique cluster_ids in DataFrame: {n_clusters_df}")

df = merge_clusters(df)

df = df.drop(columns=["mol", "cluster_id"])
df = df.rename(columns={"merged_cluster": "cluster_id"})
df.to_csv('df_fin.csv')
