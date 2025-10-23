import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore")

from model_definitions import *

class ToxScanGraphDataset(Dataset):
    def __init__(self, df, exclude_cluster_id=None):
        if exclude_cluster_id is not None:
            df = df[df["cluster_id"] != exclude_cluster_id].reset_index(drop=True)
        self.smiles = df["smiles"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        try:
            graph = mol_to_graph_cached(smi)
        except Exception as e:
            print(f"[WARN] Failed to process SMILES at idx {idx}: {smi}\n{e}")
            graph = mol_to_graph("C")
        return graph, y

def train_model(train_df, val_df, model_path, device, args, dictionary):
    train_dataset = ToxScanGraphDataset(train_df)
    val_dataset = ToxScanGraphDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_graphs, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_graphs, num_workers=0, pin_memory=True)

    labels_tensor = torch.tensor(train_dataset.labels)
    ratio_pos = (labels_tensor == 1).sum().item()
    ratio_neg = (labels_tensor == 0).sum().item()
    if ratio_pos == 0 or ratio_neg == 0:
        pos_weight = torch.tensor([1.0])
    else:
        pos_weight = torch.tensor([ratio_neg / ratio_pos])
    pos_weight = pos_weight.to(device)

    model = ToxScanModel(args, dictionary)
    model.classification_heads.clear()
    model.register_classification_head("dili", num_classes=1, inner_dim=256)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    EPOCHS = 10
    best_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch, labels in tqdm(train_loader, desc=f"Train {os.path.basename(model_path)} Epoch {epoch+1}", leave=False):
            for k in batch:
                batch[k] = batch[k].to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.cuda.amp.autocast():
                logits, *_ = model(
                    batch['src_tokens'],
                    batch['src_distance'],
                    batch['src_coord'],
                    batch['src_edge_type'],
                    batch['src_atom_feature0'],
                    batch['src_atom_feature1'],
                    batch['src_bond_feature0'],
                    batch['src_bond_feature1'],
                    batch['src_bond_feature2'],
                    batch['fingerprint'],
                    classification_head_name="dili"
                )
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch, labels in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                labels = labels.to(device).unsqueeze(1)
                with torch.cuda.amp.autocast():
                    logits, *_ = model(
                        batch['src_tokens'],
                        batch['src_distance'],
                        batch['src_coord'],
                        batch['src_edge_type'],
                        batch['src_atom_feature0'],
                        batch['src_atom_feature1'],
                        batch['src_bond_feature0'],
                        batch['src_bond_feature1'],
                        batch['src_bond_feature2'],
                        batch['fingerprint'],
                        classification_head_name="dili"
                    )
                preds = torch.sigmoid(logits).detach().cpu()
                all_preds.append(preds)
                all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)

    print(f"Saved best model: {model_path} (Val AUC: {best_auc:.4f})")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/df_fin.csv")
    print(f"Total samples: {len(df)}")
    print(f"Clusters: {sorted(df['cluster_id'].unique())}")

    scaffold_to_clusters = {}
    for _, row in df.iterrows():
        s = row["scaffold"]
        cid = row["cluster_id"]
        if s not in scaffold_to_clusters:
            scaffold_to_clusters[s] = set()
        scaffold_to_clusters[s].add(cid)

    with open("scaffold_to_clusters.pkl", "wb") as f:
        pickle.dump(scaffold_to_clusters, f)

    dictionary = DummyDict(0, 300)
    args = AttrDict({
        "encoder_layers": 6,
        "encoder_embed_dim": 512,
        "encoder_ffn_embed_dim": 256,
        "encoder_attention_heads": 4,
        "dropout": 0.3,
        "emb_dropout": 0.4,
        "attention_dropout": 0.2,
        "activation_dropout": 0.1,
        "pooler_dropout": 0.1,
        "max_seq_len": 128,
        "activation_fn": "gelu",
        "pooler_activation_fn": "tanh",
        "post_ln": False,
        "masked_token_loss": -1.0,
        "masked_coord_loss": -1.0,
        "masked_dist_loss": -1.0,
        "x_norm_loss": -1.0,
        "delta_pair_repr_norm_loss": -1.0,
    })

    train_full, val_full = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

    os.makedirs("models", exist_ok=True)
    print("Training full model...")
    train_model(
        train_df=train_full,
        val_df=val_full,
        model_path="models/model_full.pth",
        device=device,
        args=args,
        dictionary=dictionary
    )

    unique_clusters = sorted(df["cluster_id"].unique())
    print(f"Training {len(unique_clusters)} leave-one-cluster-out models...")

    for cid in unique_clusters:
        print(f"\nTraining model excluding cluster {cid}...")
        train_cid = train_full[train_full["cluster_id"] != cid].reset_index(drop=True)
        val_cid = val_full[val_full["cluster_id"] != cid].reset_index(drop=True)
        if len(train_cid) == 0 or len(val_cid) == 0:
            print(f"⚠️ Skipping cluster {cid} (not enough data)")
            continue

        train_model(
            train_df=train_cid,
            val_df=val_cid,
            model_path=f"models/model_excl_cluster_{cid}.pth",
            device=device,
            args=args,
            dictionary=dictionary
        )

    print("All models trained and saved!")