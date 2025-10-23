import hashlib
import math
import logging
import numpy as np
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pickle
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
@register_model("ToxScan")
class ToxScanModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.cls_idx = CLS_TOKEN_ID
        
        self.atom_charge_pad_idx = 0
        self.atom_H_pad_idx = 6
        self.bond_pad_idx = 8
        
        self.atom_charge_mask_idx = 9
        self.atom_H_mask_idx = 10
        self.bond_mask_idx = 12
        
        self.regression_num = 5
        
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        
        self._num_updates = None
        vocab_size = max(len(dictionary), CLS_TOKEN_ID + 1)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        
        self.bond_feature0_embed_tokens = nn.Embedding(
            self.bond_mask_idx + 3, args.encoder_attention_heads, self.bond_pad_idx
        )
        self.bond_feature1_embed_tokens = nn.Embedding(
            self.bond_mask_idx + 4, args.encoder_attention_heads, self.bond_pad_idx
        )
        self.bond_feature2_embed_tokens = nn.Embedding(
            self.bond_mask_idx + 5, args.encoder_attention_heads, self.bond_pad_idx
        )
        
        self.fingerprint_proj = NonLinearHead(
            1024, args.encoder_embed_dim, args.activation_fn
        )
        
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        
        self.classification_heads = nn.ModuleDict()           
        
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_atom_feature0,
        src_atom_feature1,
        src_bond_feature0,
        src_bond_feature1,
        src_bond_feature2,
        fingerprint,
        classification_head_name="dili",
    ):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(src_tokens)

        bond_feature0 = self.bond_feature0_embed_tokens(src_bond_feature0)
        bond_feature1 = self.bond_feature1_embed_tokens(src_bond_feature1)
        bond_feature2 = self.bond_feature2_embed_tokens(src_bond_feature2)

        gbf_feature = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result + bond_feature0 + bond_feature1 + bond_feature2
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous().view(-1, src_tokens.size(1), src_tokens.size(1))

        encoder_rep, _, _, _, _ = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)

        encoder_rep[:, 0, :] = encoder_rep[:, 0, :] + self.fingerprint_proj(fingerprint.to(encoder_rep.dtype))

        logits = self.classification_heads[classification_head_name](encoder_rep)
        return logits, 
         

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]       
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MyClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]        
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("ToxScan", "ToxScan")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)


@register_model_architecture("ToxScan", "ToxScan_base")
def ToxScan_base_architecture(args):
    base_architecture(args)


class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](
                x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
            )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)
        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm

MAX_ATOMS = 128
CLS_TOKEN_ID = 118

def mol_to_graph_cached(smiles, cache_dir="cache_graphs"):
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(smiles.encode()).hexdigest()
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    g = mol_to_graph(smiles)
    with open(path, "wb") as f:
        pickle.dump(g, f)
    return g


def mol_to_graph(smiles, num_confs=11, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        randomSeed=seed,
        useRandomCoords=True,
        useBasicKnowledge=False,
        enforceChirality=False,
        clearConfs=True,
        numThreads=0,
    )

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed)

    conf_id = np.random.randint(mol.GetNumConformers())
    conf = mol.GetConformer(conf_id)

    atoms = list(mol.GetAtoms())
    n_atoms = min(len(atoms), MAX_ATOMS)

    total_len = n_atoms + 1
    src_tokens = np.full((MAX_ATOMS + 1,), 0, dtype=np.int64)
    src_atom_feature0 = np.full((MAX_ATOMS + 1,), 0, dtype=np.int64)
    src_atom_feature1 = np.full((MAX_ATOMS + 1,), 0, dtype=np.int64)

    src_tokens[0] = CLS_TOKEN_ID
    src_atom_feature0[0] = 0
    src_atom_feature1[0] = 0

    atomic_nums = [atom.GetAtomicNum() for atom in atoms[:n_atoms]]
    charges = [atom.GetFormalCharge() for atom in atoms[:n_atoms]]
    hcounts = [atom.GetTotalNumHs() for atom in atoms[:n_atoms]]

    src_tokens[1:1 + n_atoms] = atomic_nums
    src_atom_feature0[1:1 + n_atoms] = charges
    src_atom_feature1[1:1 + n_atoms] = hcounts

    coords = np.zeros((MAX_ATOMS + 1, 3), dtype=np.float32)
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(atoms[i].GetIdx())
        coords[1 + i] = [pos.x, pos.y, pos.z]

    if n_atoms > 0:
        coords[1:1 + n_atoms] -= coords[1:1 + n_atoms].mean(axis=0, keepdims=True)
        scale = np.linalg.norm(coords[1:1 + n_atoms], axis=1).max() + 1e-6
        coords[1:1 + n_atoms] /= scale

    src_coord = coords

    coords_t = torch.tensor(coords[:total_len], dtype=torch.float32)
    dist = torch.zeros((MAX_ATOMS + 1, MAX_ATOMS + 1), dtype=torch.float32)
    dist[:total_len, :total_len] = torch.cdist(coords_t, coords_t, p=2)
    src_distance = dist

    edge_type = np.zeros((MAX_ATOMS + 1, MAX_ATOMS + 1), dtype=np.int64)
    arom = np.zeros((MAX_ATOMS + 1, MAX_ATOMS + 1), dtype=np.int64)
    conj = np.zeros((MAX_ATOMS + 1, MAX_ATOMS + 1), dtype=np.int64)
    ring = np.zeros((MAX_ATOMS + 1, MAX_ATOMS + 1), dtype=np.int64)

    atom_indices = {a.GetIdx(): i + 1 for i, a in enumerate(atoms[:n_atoms])}  # +1 из-за [CLS]

    for bond in mol.GetBonds():
        i_old, j_old = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i_old in atom_indices and j_old in atom_indices:
            i, j = atom_indices[i_old], atom_indices[j_old]
            btype = int(bond.GetBondTypeAsDouble())
            edge_type[i, j] = edge_type[j, i] = btype
            arom[i, j] = arom[j, i] = int(bond.GetIsAromatic())
            conj[i, j] = conj[j, i] = int(bond.GetIsConjugated())
            ring[i, j] = ring[j, i] = int(bond.IsInRing())

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros((1024,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    return dict(
        src_tokens=torch.tensor(src_tokens, dtype=torch.long),
        src_distance=torch.tensor(src_distance, dtype=torch.float32),
        src_coord=torch.tensor(src_coord, dtype=torch.float32),
        src_edge_type=torch.tensor(edge_type, dtype=torch.long),
        src_atom_feature0=torch.tensor(src_atom_feature0, dtype=torch.long),
        src_atom_feature1=torch.tensor(src_atom_feature1, dtype=torch.long),
        src_bond_feature0=torch.tensor(arom, dtype=torch.long),
        src_bond_feature1=torch.tensor(conj, dtype=torch.long),
        src_bond_feature2=torch.tensor(ring, dtype=torch.long),
        fingerprint=torch.tensor(fp_arr, dtype=torch.float32),
    )
    
class DummyDict:
    def __init__(self, pad_idx=0, n_tokens=300):
        self.pad_idx = pad_idx
        self._len = n_tokens
    def pad(self): return self.pad_idx
    def __len__(self): return self._len

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def collate_graphs(batch):
    graphs, labels = zip(*batch)
    out = {key: torch.stack([g[key] for g in graphs]) for key in graphs[0].keys()}
    labels = torch.stack(labels)
    return out, labels

def get_scaffold(smiles):
    """Получить Bemis-Murcko scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = AllChem.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except:
        return ""