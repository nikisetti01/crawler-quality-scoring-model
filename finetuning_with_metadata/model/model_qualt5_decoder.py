import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

QUALT5_MODEL = "pyterrier-quality/qt5-small"
TOKENIZER = AutoTokenizer.from_pretrained(QUALT5_MODEL)

# Cache una volta per tutte per evitare pesi creati ad ogni import
_T5 = AutoModelForSeq2SeqLM.from_pretrained(QUALT5_MODEL)
HIDDEN_SIZE = _T5.config.d_model  # ~512

class MetadataEncoder(nn.Module):
    """
    Encoda ancore, domini e numerici in 5 'meta token' (anchor_out, anchor_in, domain_out, domain_in, numerics)
    ciascuno di dimensione hidden_dim.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Embedding "povero" per gli anchor token ids: media sui token poi proiezione
        self.anchor_embedding = nn.Embedding(len(TOKENIZER), 32)
        self.anchor_proj = nn.Sequential(nn.Linear(32, hidden_dim), nn.ReLU())

        # Domini già ridotti a 64 in input (via embedding separata nel modello)
        self.domain_proj  = nn.Sequential(nn.Linear(64, hidden_dim), nn.ReLU())
        self.numeric_proj = nn.Sequential(nn.Linear(8,  hidden_dim), nn.ReLU())

    def forward(self, anchor_out_ids, anchor_in_ids, domain_out, domain_in, numerics):
        # anchor_*_ids: [B, La] token ids -> mean pooling sull'asse token
        anchor_out = self.anchor_proj(self.anchor_embedding(anchor_out_ids).mean(dim=1))  # [B, Hm]
        anchor_in  = self.anchor_proj(self.anchor_embedding(anchor_in_ids).mean(dim=1))   # [B, Hm]
        domain_out = self.domain_proj(domain_out.float())                                 # [B, Hm]
        domain_in  = self.domain_proj(domain_in.float())                                   # [B, Hm]
        numerics   = self.numeric_proj(numerics)                                          # [B, Hm]
        # stack -> [B, 5, Hm]
        return torch.stack([anchor_out, anchor_in, domain_out, domain_in, numerics], dim=1)


class AdditiveUniAttention(nn.Module):
    """
    Q dai meta-token [B, M, Hm], K/V dall'output encoder completo [B, L, He].
    Additive attention (Bahdanau): non richiede dim uguali Q/K, usa mask dell'encoder.
    """
    def __init__(self, meta_h, txt_h, attn_h=64, dropout=0.1):
        super().__init__()
        self.Wq = nn.Linear(meta_h, attn_h, bias=True)
        self.Wk = nn.Linear(txt_h,  attn_h, bias=True)
        self.v  = nn.Linear(attn_h, 1,      bias=True)
        self.Vv = nn.Linear(txt_h,  meta_h, bias=True)  # proietta i Values nello spazio meta
        self.norm = nn.LayerNorm(meta_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, meta_tokens, text_tokens, attention_mask):
        """
        meta_tokens:  [B, M, Hm]
        text_tokens:  [B, L, He]
        attention_mask: [B, L] con 1=token valido, 0=pad
        """
        B, M, Hm = meta_tokens.shape
        L = text_tokens.size(1)

        Qe = self.Wq(meta_tokens)         # [B, M, A]
        Ke = self.Wk(text_tokens)         # [B, L, A]

        # Broadcasting per additive score: v^T tanh(Qe_i + Ke_j)
        Qe_exp = Qe.unsqueeze(2).expand(-1, -1, L, -1)   # [B, M, L, A]
        Ke_exp = Ke.unsqueeze(1).expand(-1, M, -1, -1)   # [B, M, L, A]
        e_ij = torch.tanh(Qe_exp + Ke_exp)               # [B, M, L, A]
        scores = self.v(e_ij).squeeze(-1)                # [B, M, L]

        if attention_mask is not None:
            mask = (attention_mask == 1).unsqueeze(1).expand(-1, M, -1)  # [B, M, L]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)             # [B, M, L]

        Vproj = self.Vv(text_tokens)                     # [B, L, Hm]
        context = torch.bmm(attn, Vproj)                 # [B, M, Hm]

        return self.norm(meta_tokens + self.dropout(context))  # [B, M, Hm]


class MultiModalWebClassifier(nn.Module):
    """
    QualT5 decoder-style:
      - encoder (T5) sul testo
      - meta -> Additive Uni-Attn (K/V da tutta la seq encoder + mask)
      - il vettore fuso -> mappato a un prompt embedding per il decoder
      - training 1-step con labels 'true'/'false' (senza cambiare la tua pipeline)
    """
    def __init__(self, hidden_dim=128, prompt_length=1):
        super().__init__()
        self.model = _T5
        self.tokenizer = TOKENIZER

        # semplice embedding per domini su vocab T5 (input: ids [B, Ld])
        self.domains_embedding = nn.Embedding(len(TOKENIZER), 64)

        self.meta_encoder = MetadataEncoder(hidden_dim)
        self.uni_attn     = AdditiveUniAttention(meta_h=hidden_dim, txt_h=HIDDEN_SIZE, attn_h=64, dropout=0.1)

        self.prompt_mapper = nn.Sequential(
            nn.Linear(hidden_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Per il tuo training 'true/false' 1‑token meglio tenerlo a 1
        self.prompt_length = prompt_length

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_out_mask,
                anchor_in_ids,  anchor_in_mask,
                domains_out_ids, domains_in_ids,
                numerics, labels=None):
        """
        labels: tensor di token id target (es. 'true'/'false' senza EOS), shape [B, 1]
        """

        # 1) Encode testo (T5 encoder)
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hidden = enc_out.last_hidden_state               # [B, L, He]

        # 2) Encode domini (masked mean semplificata già fatta con mean)
        dom_out = self.domains_embedding(domains_out_ids).mean(dim=1)  # [B, 64]
        dom_in  = self.domains_embedding(domains_in_ids).mean(dim=1)   # [B, 64]

        # 3) Meta tokens + Uni-Attn (K,V dalla sequenza intera + attention_mask)
        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, dom_out, dom_in, numerics)  # [B, 5, Hm]
        uni_out = self.uni_attn(meta_tokens, enc_hidden, attention_mask)                           # [B, 5, Hm]
        meta_vec = uni_out.mean(dim=1)                                                             # [B, Hm]

        # 4) Prompt per il decoder (embedding diretto)
        prompt = self.prompt_mapper(meta_vec)                           # [B, He]
        prompt = prompt.unsqueeze(1).expand(-1, self.prompt_length, -1) # [B, Tprompt, He]

        # 5) Decoder (usa encoder_outputs corretti)
        outputs = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            decoder_inputs_embeds=prompt,
            labels=labels,                  # [B, 1] se prompt_length=1
            return_dict=True
        )
        return outputs
