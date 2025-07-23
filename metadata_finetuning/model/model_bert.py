
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from datasets import load_from_disk, concatenate_datasets, ClassLabel
from tqdm import tqdm
import wandb
import os
# === MODEL ===
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MetadataEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.anchor_encoder = nn.Sequential(nn.Linear(32, hidden_dim), nn.ReLU())
        self.domain_proj = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        self.numeric_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())

    def forward(self, anchor_out, anchor_in, domain_out, domain_in, numerics):
        return torch.stack([
            self.anchor_encoder(anchor_out.float()),
            self.anchor_encoder(anchor_in.float()),
            self.domain_proj(domain_out.float()),
            self.domain_proj(domain_in.float()),
            self.numeric_proj(numerics)
        ], dim=1)

class UniAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(768, hidden_dim)
        self.v_proj = nn.Linear(768, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, metadata_tokens, bert_tokens):
        Q = self.q_proj(metadata_tokens)
        K = self.k_proj(bert_tokens)
        V = self.v_proj(bert_tokens)
        attended, _ = self.attn(Q, K, V)
        return self.norm(metadata_tokens + self.dropout(attended))

class MultiModalWebClassifier(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.domains_embedding = nn.Embedding(30522, 64)
        self.meta_encoder = MetadataEncoder(hidden_dim)
        self.uni_attn = UniAttention(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 768, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_out_mask,
                anchor_in_ids, anchor_in_mask,
                domains_out_ids, domains_in_ids,
                numerics):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0]
        bert_tokens = bert_out.last_hidden_state

        domain_out_embed = self.domains_embedding(domains_out_ids).mean(dim=1)
        domain_in_embed = self.domains_embedding(domains_in_ids).mean(dim=1)

        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, domain_out_embed, domain_in_embed, numerics)
        uni_out = self.uni_attn(meta_tokens, bert_tokens)
        meta_vec = uni_out.mean(dim=1)

        combined = torch.cat([bert_cls, meta_vec], dim=-1)
        return torch.sigmoid(self.head(combined)).squeeze(-1)