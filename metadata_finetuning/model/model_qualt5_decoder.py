import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

QUALT5_MODEL = "pyterrier-quality/qt5-small"
TOKENIZER = AutoTokenizer.from_pretrained(QUALT5_MODEL)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(QUALT5_MODEL)
ENCODER = MODEL.encoder
DECODER = MODEL.decoder
HIDDEN_SIZE = ENCODER.config.hidden_size  # ~512

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
        self.k_proj = nn.Linear(HIDDEN_SIZE, hidden_dim)
        self.v_proj = nn.Linear(HIDDEN_SIZE, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, metadata_tokens, text_tokens):
        Q = self.q_proj(metadata_tokens)
        K = self.k_proj(text_tokens)
        V = self.v_proj(text_tokens)
        attended, _ = self.attn(Q, K, V)
        return self.norm(metadata_tokens + self.dropout(attended))

class MultiModalWebClassifier(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(QUALT5_MODEL)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.tokenizer = TOKENIZER
        self.domains_embedding = nn.Embedding(len(self.tokenizer), 64)
        self.meta_encoder = MetadataEncoder(hidden_dim)
        self.uni_attn = UniAttention(hidden_dim)
        self.combined_proj = nn.Sequential(
            nn.Linear(hidden_dim + HIDDEN_SIZE, hidden_dim + HIDDEN_SIZE),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim + HIDDEN_SIZE),
            nn.Dropout(0.1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_out_mask,
                anchor_in_ids, anchor_in_mask,
                domains_out_ids, domains_in_ids,
                numerics):

        # === Encoder ===
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = enc_out.last_hidden_state

        # === Decoder (usando stesso testo come decoder input) ===
        decoder_out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )
        decoder_cls = decoder_out.last_hidden_state[:, 0]  # "CLS" del decoder

        # === Metadata path ===
        domain_out_embed = self.domains_embedding(domains_out_ids).mean(dim=1)
        domain_in_embed = self.domains_embedding(domains_in_ids).mean(dim=1)
        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, domain_out_embed, domain_in_embed, numerics)
        uni_out = self.uni_attn(meta_tokens, encoder_hidden_states)
        meta_vec = uni_out.mean(dim=1)

        # === Fusione e output ===
        combined = torch.cat([decoder_cls, meta_vec], dim=-1)
        combined = self.combined_proj(combined)
        return torch.sigmoid(self.head(combined)).squeeze(-1)
