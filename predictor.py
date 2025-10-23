import os
import pickle
from tqdm import tqdm
import torch
from model_definitions import (
    ToxScanModel,
    mol_to_graph_cached,
    get_scaffold,
    DummyDict,
    AttrDict
)

def get_model_args():
    return AttrDict({
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

def load_unique_clusters():
    with open("scaffold_to_clusters.pkl", "rb") as f:
        scaffold_to_clusters = pickle.load(f)
    all_clusters = set()
    for clusts in scaffold_to_clusters.values():
        all_clusters.update(clusts)
    return sorted(all_clusters)

def predict_with_ensemble(smiles_list, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("scaffold_to_clusters.pkl", "rb") as f:
        scaffold_to_clusters = pickle.load(f)

    args = get_model_args()
    dictionary = DummyDict(0, 300)

    print("Loading full model...")
    full_model = ToxScanModel(args, dictionary)
    full_model.classification_heads.clear()
    full_model.register_classification_head("dili", num_classes=1, inner_dim=256)
    full_model.load_state_dict(torch.load("models/model_full.pth", map_location=device))
    full_model.to(device)
    full_model.eval()

    print("Loading cluster-specific models...")
    unique_clusters = load_unique_clusters()
    cluster_models = {}
    for cid in unique_clusters:
        path = f"models/model_excl_cluster_{cid}.pth"
        if os.path.exists(path):
            model = ToxScanModel(args, dictionary)
            model.classification_heads.clear()
            model.register_classification_head("dili", num_classes=1, inner_dim=256)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            cluster_models[cid] = model

    print(f"Loaded {len(cluster_models)} cluster models + 1 full model.")

    preds = []
    for smi in tqdm(smiles_list, desc="Predicting"):
        scaffold = get_scaffold(smi)
        relevant_clusters = scaffold_to_clusters.get(scaffold, set())

        if len(relevant_clusters) == 0:
            model_to_use = full_model
        elif len(relevant_clusters) == 1:
            cid = list(relevant_clusters)[0]
            model_to_use = cluster_models.get(cid, full_model)
        else:
            logits_list = []
            for cid in relevant_clusters:
                if cid in cluster_models:
                    graph = mol_to_graph_cached(smi)
                    batch = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        logits, *_ = cluster_models[cid](
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
                    logits_list.append(logits)
            if logits_list:
                avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                pred = torch.sigmoid(avg_logits).item()
                preds.append(pred)
                continue
            else:
                model_to_use = full_model

        graph = mol_to_graph_cached(smi)
        batch = {k: v.unsqueeze(0).to(device) for k, v in graph.items()}
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits, *_ = model_to_use(
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
        pred = torch.sigmoid(logits).item()
        preds.append(pred)

    return preds