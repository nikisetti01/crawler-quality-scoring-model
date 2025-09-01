import torch
import torch.nn as nn
from transformers import BertModel

HIDDEN_DIM = 128
BERT_H = 768  # hidden size BERT base


class MetadataEncoder(nn.Module):
    """
    Encode heterogeneous metadata sources (anchors, domains, numerics)
    into a unified hidden representation. Each feature type is projected
    through a small feed-forward layer into the same hidden space.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.anchor_encoder = nn.Sequential(nn.Linear(32, hidden_dim), nn.ReLU())
        self.domain_proj    = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        self.numeric_proj   = nn.Sequential(nn.Linear(8,  hidden_dim), nn.ReLU())

    def forward(self, anchor_out, anchor_in, domain_out, domain_in, numerics):
        # Stack different metadata encodings along a pseudo-sequence axis
        return torch.stack([
            self.anchor_encoder(anchor_out.float()),   # outgoing anchors
            self.anchor_encoder(anchor_in.float()),    # incoming anchors
            self.domain_proj(domain_out.float()),      # outgoing domains
            self.domain_proj(domain_in.float()),       # incoming domains
            self.numeric_proj(numerics)                # numerical link stats
        ], dim=1)  # shape [B, 5, H]


class AdditiveUniAttention(nn.Module):
    """
    Unified additive attention (Bahdanau-style) that aligns metadata tokens
    with the textual representation. Each metadata token queries the entire
    text sequence to extract relevant contextual information.
    """
    def __init__(self, meta_h=HIDDEN_DIM, txt_h=BERT_H, attn_h=64):
        super().__init__()
        self.Wq = nn.Linear(meta_h, attn_h, bias=True)   # project metadata tokens
        self.Wk = nn.Linear(txt_h, attn_h, bias=True)    # project textual tokens
        self.v  = nn.Linear(attn_h, 1, bias=True)        # scoring vector
        self.Vv = nn.Linear(txt_h, meta_h, bias=True)    # project values into meta space
        self.norm = nn.LayerNorm(meta_h)
        self.dropout = nn.Dropout(0.1)

    def forward(self, meta_tokens, bert_hidden, bert_attn_mask):
        """
        meta_tokens: metadata sequence [B, M, H]
        bert_hidden: contextual text embeddings [B, L, BERT_H]
        bert_attn_mask: mask for text padding [B, L]
        return: metadata tokens enriched with textual context [B, M, H]
        """
        B, M, H = meta_tokens.size()
        L = bert_hidden.size(1)

        # Linear projections
        Qe = self.Wq(meta_tokens)    # queries from metadata
        Ke = self.Wk(bert_hidden)    # keys from text

        # Compute additive attention scores: v^T tanh(Q + K)
        Qe_exp = Qe.unsqueeze(2).expand(-1, -1, L, -1)  # [B, M, L, A]
        Ke_exp = Ke.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, L, A]
        e_ij = torch.tanh(Qe_exp + Ke_exp)
        scores = self.v(e_ij).squeeze(-1)               # [B, M, L]

        # Apply mask to ignore padding tokens
        if bert_attn_mask is not None:
            mask = (bert_attn_mask == 1).unsqueeze(1).expand(-1, M, -1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Normalize into attention weights
        attn = torch.softmax(scores, dim=-1)            # [B, M, L]

        # Weighted sum of textual values projected into metadata space
        Vproj = self.Vv(bert_hidden)                    # [B, L, H]
        context = torch.bmm(attn, Vproj)                # [B, M, H]

        # Residual connection + layer normalization
        out = self.norm(meta_tokens + self.dropout(context))
        return out


class MultiModalWebClassifier(nn.Module):
    """
    A multimodal classifier that integrates:
      1. Textual representations from a pre-trained Transformer (BERT).
      2. Structural/contextual metadata (anchors, domains, numeric stats).
      3. A unified attention mechanism to let metadata tokens attend over text.
    Final prediction is binary relevance of a candidate link.
    """
    def __init__(self, encoder: BertModel, hidden_dim=HIDDEN_DIM, use_lora=False):
        super().__init__()
        self.encoder = encoder

        # Domain-level categorical metadata embedded in the same space
        self.domains_embedding = nn.Embedding(30522, 64)

        # Metadata encoder + cross-modal attention
        self.meta_encoder = MetadataEncoder(hidden_dim)
        self.uni_attn     = AdditiveUniAttention(meta_h=hidden_dim, txt_h=BERT_H, attn_h=64)

        # Classification head: combines text [CLS] + enriched metadata
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + BERT_H, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # binary logit
        )

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_in_ids,
                anchor_out_mask, anchor_in_mask,
                domains_out_ids, domains_in_ids,
                numerics):

        # (1) Encode textual content with Transformer
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = enc.last_hidden_state               # contextualized sequence [B, L, 768]
        cls = last_hidden[:, 0]                           # [CLS] token embedding [B, 768]

        # (2) Encode domain lists with masked mean pooling
        def masked_mean(emb, ids):
            mask = (ids != 0).unsqueeze(-1)
            emb = emb * mask
            return emb.sum(1) / mask.sum(1).clamp(min=1e-6)

        dom_out = masked_mean(self.domains_embedding(domains_out_ids), domains_out_ids)
        dom_in  = masked_mean(self.domains_embedding(domains_in_ids),  domains_in_ids)

        # (3) Encode metadata and refine with cross-modal attention
        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, dom_out, dom_in, numerics)  # [B, 5, H]
        attn_out = self.uni_attn(meta_tokens, last_hidden, attention_mask)                         # [B, 5, H]
        meta_vec = attn_out.mean(dim=1)                                                            # aggregate [B, H]

        # (4) Fuse CLS + metadata vector → classification head
        combined = torch.cat([cls, meta_vec], dim=-1)  # [B, 768+H]
        logits = self.head(combined).squeeze(-1)       # [B]
        return logits
