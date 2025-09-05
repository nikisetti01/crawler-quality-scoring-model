import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

QUALT5_MODEL = "pyterrier-quality/qt5-small"
TOKENIZER = AutoTokenizer.from_pretrained(QUALT5_MODEL)

# Cache model weights once to avoid re-loading at each import
_T5 = AutoModelForSeq2SeqLM.from_pretrained(QUALT5_MODEL)
HIDDEN_SIZE = _T5.config.d_model  # ≈512 for qt5-small


class MetadataEncoder(nn.Module):
    """
    Encodes heterogeneous metadata into a compact sequence of 5 'meta tokens':
      - outgoing anchors
      - incoming anchors
      - outgoing domains
      - incoming domains
      - numeric statistics
    Each component is projected into a common hidden_dim space so they can
    later interact with the text encoder representations.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Anchor tokens: simple learned embeddings → mean pooling → projection
        self.anchor_embedding = nn.Embedding(len(TOKENIZER), 32)
        self.anchor_proj = nn.Sequential(nn.Linear(32, hidden_dim), nn.ReLU())

        # Domains are already reduced to 64-dim → projection to hidden_dim
        self.domain_proj  = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        # Numeric features (8-dim vector) → projection
        self.numeric_proj = nn.Sequential(nn.Linear(8,  hidden_dim), nn.ReLU())

    def forward(self, anchor_out_ids, anchor_in_ids, domain_out, domain_in, numerics):
        # Mean pooling over anchor tokens
        anchor_out = self.anchor_proj(self.anchor_embedding(anchor_out_ids).mean(dim=1))  # [B, Hm]
        anchor_in  = self.anchor_proj(self.anchor_embedding(anchor_in_ids).mean(dim=1))   # [B, Hm]

        # Domain projections
        domain_out = self.domain_proj(domain_out.float())  # [B, Hm]
        domain_in  = self.domain_proj(domain_in.float())   # [B, Hm]

        # Numeric stats projection
        numerics   = self.numeric_proj(numerics)           # [B, Hm]

        # Stack into a meta-sequence of 5 tokens
        return torch.stack([anchor_out, anchor_in, domain_out, domain_in, numerics], dim=1)  # [B, 5, Hm]


class AdditiveUniAttention(nn.Module):
    """
    Unified additive attention (Bahdanau-style):
      - Queries come from metadata tokens [B, M, Hm].
      - Keys/Values come from the full text encoder sequence [B, L, He].
    This mechanism aligns metadata with relevant parts of the textual context.
    """
    def __init__(self, meta_h, txt_h, attn_h=64, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(meta_h, attn_h, bias=True)   # map meta → attn space
        self.Wk = nn.Linear(txt_h,  attn_h, bias=True)   # map text → attn space
        self.v  = nn.Linear(attn_h, 1, bias=True)        # scoring vector
        self.Vv = nn.Linear(txt_h,  meta_h, bias=True)   # project values into meta space
        self.norm = nn.LayerNorm(meta_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, meta_tokens, text_tokens, attention_mask):
        """
        meta_tokens:  metadata tokens [B, M, Hm]
        text_tokens:  encoder hidden states [B, L, He]
        attention_mask: binary mask for padding [B, L]
        """
        B, M, Hm = meta_tokens.shape
        L = text_tokens.size(1)

        # Linear projections
        Qe = self.Wq(meta_tokens)    # [B, M, A]
        Ke = self.Wk(text_tokens)    # [B, L, A]

        # Compute additive attention scores: v^T tanh(Q + K)
        Qe_exp = Qe.unsqueeze(2).expand(-1, -1, L, -1)   # [B, M, L, A]
        Ke_exp = Ke.unsqueeze(1).expand(-1, M, -1, -1)   # [B, M, L, A]
        e_ij = torch.tanh(Qe_exp + Ke_exp)
        scores = self.v(e_ij).squeeze(-1)                # [B, M, L]

        # Mask out padding tokens
        if attention_mask is not None:
            mask = (attention_mask == 1).unsqueeze(1).expand(-1, M, -1)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Normalize attention
        attn = torch.softmax(scores, dim=-1)             # [B, M, L]

        # Weighted sum of values projected into meta space
        Vproj = self.Vv(text_tokens)                     # [B, L, Hm]
        context = torch.bmm(attn, Vproj)                 # [B, M, Hm]

        # Residual connection + normalization
        return self.norm(meta_tokens + self.dropout(context))  # [B, M, Hm]


class MultiModalWebClassifier(nn.Module):
    """
    Multimodal architecture built on top of QualT5:
      1. Text encoder (T5) processes the raw page text.
      2. Metadata encoder produces 5 meta tokens.
      3. Additive UniAttention lets metadata attend over the text sequence.
      4. The enriched metadata vector is mapped into a prompt embedding.
      5. Decoder is conditioned on this prompt to predict a binary label
         ("true"/"false") in a seq2seq manner.
    This formulation keeps the training in the native T5 pipeline
    while injecting structured metadata via prompts.
    """
    def __init__(self, hidden_dim=128, prompt_length=1):
        super().__init__()
        self.model = _T5
        self.tokenizer = TOKENIZER

        # Simple embedding for domain categorical features
        self.domains_embedding = nn.Embedding(len(TOKENIZER), 64)

        # Metadata encoder + cross-modal attention
        self.meta_encoder = MetadataEncoder(hidden_dim)
        self.uni_attn     = AdditiveUniAttention(meta_h=hidden_dim, txt_h=HIDDEN_SIZE, attn_h=64, dropout=0.1)

        # Project meta representation into T5 hidden space (prompt embedding)
        self.prompt_mapper = nn.Sequential(
            nn.Linear(hidden_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.prompt_length = prompt_length  # length of prompt fed to decoder

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_out_mask,
                anchor_in_ids,  anchor_in_mask,
                domains_out_ids, domains_in_ids,
                numerics, labels=None):
        """
        Forward pass:
          input_ids / attention_mask : text inputs
          anchor/domain/numerics     : metadata features
          labels                     : single-token target ("true"/"false")
        """
        # (1) Encode text with T5 encoder
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hidden = enc_out.last_hidden_state               # [B, L, He]

        # (2) Encode domains (mean pooling over domain IDs)
        dom_out = self.domains_embedding(domains_out_ids).mean(dim=1)  # [B, 64]
        dom_in  = self.domains_embedding(domains_in_ids).mean(dim=1)   # [B, 64]

        # (3) Encode metadata and enrich with cross-modal attention
        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, dom_out, dom_in, numerics)  # [B, 5, Hm]
        uni_out = self.uni_attn(meta_tokens, enc_hidden, attention_mask)                           # [B, 5, Hm]
        meta_vec = uni_out.mean(dim=1)                                                             # [B, Hm]

        # (4) Map enriched metadata into a decoder prompt
        prompt = self.prompt_mapper(meta_vec)                           # [B, He]
        prompt = prompt.unsqueeze(1).expand(-1, self.prompt_length, -1) # [B, Tprompt, He]

        # (5) Run decoder conditioned on encoder outputs + metadata prompt
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            decoder_inputs_embeds=prompt,
            labels=labels,                  # target: [B, 1] (true/false)
            return_dict=True
        )
        return outputs
