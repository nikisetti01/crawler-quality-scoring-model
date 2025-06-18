# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from transformers import BertTokenizerFast, BertModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np

from datasets import Dataset as HFDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[✓] Using device: {device}")
df=pd.read_csv("third_dataset_sample.csv")

print(df.columns)
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

def preprocess_dataframe(df):
    # Parse domini (stringhe con "|") in liste
    def pipe_to_list(val):
        if isinstance(val, str):
            return val.split('|')
        return []

    for col in ['outlink_list_domains', 'inlink_list_domains']:
        df[col] = df[col].fillna('').apply(pipe_to_list)

    # Anchor text (testo breve)
    for col in ['outlink_anchors', 'inlink_anchors']:
        df[col] = df[col].fillna('').astype(str)

    # Features numeriche da normalizzare
    num_cols = ['num_outlinks', 'length_outlinks', 'outlink_domains_count', 'outlink_slashes_count']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # MultiLabelBinarizer: vocabolario domini
    mlb = MultiLabelBinarizer()
    all_domains = df['outlink_list_domains'].tolist() + df['inlink_list_domains'].tolist()
    print(all_domains[:5])  # primi 5 elementi
    print([x for x in all_domains if x])  # lista non vuota

    mlb.fit(all_domains)

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), mlb, scaler


class WebPageDataset(Dataset):
    def __init__(self, df, tokenizer, mlb, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.max_len = max_len
        self.numerical_cols = ['num_outlinks', 'length_outlinks', 
                              'outlink_domains_count', 'outlink_slashes_count']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert domains to numeric array first
        def safe_transform(feature):
            # Ensure proper list format and empty handling
            domains = row[feature] if isinstance(row[feature], list) else []
            transformed = self.mlb.transform([domains]).astype(np.float32)
            return transformed[0]  # Return 1D array

        # Text encoding
        text_enc = self.tokenizer(
            row['text'], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_len, 
            return_tensors='pt'
        )

        # Convert domains to float32 tensors
        domains_out = torch.tensor(safe_transform('outlink_list_domains'), dtype=torch.float32)
        domains_in = torch.tensor(safe_transform('inlink_list_domains'), dtype=torch.float32)

        # Numerical features
        nums = torch.tensor(row[self.numerical_cols].values.astype(np.float32), dtype=torch.float32)

        # Anchor texts
        def encode_anchor(text):
            return self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=32, 
                return_tensors='pt'
            )
            
        anchor_out = encode_anchor(row['outlink_anchors'])
        anchor_in = encode_anchor(row['inlink_anchors'])

        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'anchor_out_ids': anchor_out['input_ids'].squeeze(0),
            'anchor_out_mask': anchor_out['attention_mask'].squeeze(0),
            'anchor_in_ids': anchor_in['input_ids'].squeeze(0),
            'anchor_in_mask': anchor_in['attention_mask'].squeeze(0),
            'domains_out': domains_out,
            'domains_in': domains_in,
            'numerics': nums,
            'label': torch.tensor(row['label'], dtype=torch.float32)
        }

# %%
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.anchor_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
        )
        self.domain_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU()
        )

    def forward(self, anchor_out, anchor_in, domain_out, domain_in, numerics):
        
        

        # Encode anchor (media dei token ID come placeholder per BERT)
        anchor_out_feat = self.anchor_encoder(anchor_out.float())  # [B, hidden]
        anchor_in_feat = self.anchor_encoder(anchor_in.float())    # [B, hidden]
      
        # Encode domains (applico la media sui 32 embedding, se presenti)
        domain_out_feat = self.domain_proj(domain_out)  # [B, hidden]
        domain_in_feat = self.domain_proj(domain_in)    # [B, hidden]
  
        # Encode numeric features
        num_feat = self.numeric_proj(numerics)  # [B, hidden]
       

        # Stack tutto in una sequenza: [B, 5, hidden]
        stacked = torch.stack([
            anchor_out_feat,
            anchor_in_feat,
            domain_out_feat,#.mean(dim=0),
            domain_in_feat,#.mean(dim=0),
            num_feat,#.mean(dim=0)
        ], dim=1)

        return stacked



# %%
class UniAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bert_proj=nn.Linear(768, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, metadata_tokens, bert_tokens):
        bert_tokens = self.bert_proj(bert_tokens)
       # print(f"bert_tokens shape: {bert_tokens.shape}", flush=True)
        Q = self.q_proj(metadata_tokens)
        K = self.k_proj(bert_tokens)
        V = self.v_proj(bert_tokens)
 
        attended, _ = self.attn(Q, K, V)
        return self.norm(metadata_tokens + self.dropout(attended))

class MultiModalWebClassifier(nn.Module):
    def __init__(self, domain_dim, hidden_dim=128):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.meta_encoder = MetadataEncoder(domain_dim, hidden_dim)
        self.uni_attn = UniAttention(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask,
                anchor_out_ids, anchor_out_mask,
                anchor_in_ids, anchor_in_mask,
                domains_out, domains_in, numerics):

        
        # BERT path
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_out.last_hidden_state[:, 0]
        bert_tokens = bert_out.last_hidden_state
   
        # Metadata path
        meta_tokens = self.meta_encoder(anchor_out_ids, anchor_in_ids, domains_out, domains_in, numerics)

        # Uni-Attention
        uni_out = self.uni_attn(meta_tokens, bert_tokens)
  # [B, 5, hidden]
        meta_vec = uni_out.mean(dim=1)
       # [B, hidden]

        # Fusion + Classification
        combined = torch.cat([bert_cls, meta_vec], dim=-1)
        return torch.sigmoid(self.head(combined)).squeeze(-1)


# %%


import torch.optim as optim
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def train_phase1(model, train_loader, val_loader, epochs=5):
    # Freeze BERT parameters
    for param in model.bert.parameters():
        param.requires_grad = False
        
    # Only train metadata components
    optimizer = optim.AdamW([
        {'params': model.meta_encoder.parameters()},
        {'params': model.uni_attn.parameters()},
        {'params': model.head.parameters()}
    ], lr=1e-3)
    
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
 
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                anchor_out_ids=batch['anchor_out_ids'],
                anchor_out_mask=batch['anchor_out_mask'],  # Added
                anchor_in_ids=batch['anchor_in_ids'],
                anchor_in_mask=batch['anchor_in_mask'],    # Added
                domains_out=batch['domains_out'],
                domains_in=batch['domains_in'],
                numerics=batch['numerics']
            )
            
            loss = criterion(outputs, batch['label'])
            print(f"[Train Phase1] Batch Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
             
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    anchor_out_ids=batch['anchor_out_ids'],
                    anchor_out_mask=batch['anchor_out_mask'],  # Added
                    anchor_in_ids=batch['anchor_in_ids'],
                    anchor_in_mask=batch['anchor_in_mask'],    # Added
                    domains_out=batch['domains_out'],
                    domains_in=batch['domains_in'],
                    numerics=batch['numerics']
                )
                batch_val_loss = criterion(outputs, batch['label']).item()
                print(f"[Val Phase1] Batch Loss: {batch_val_loss:.4f}")
                val_loss += batch_val_loss

        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'phase1_best.pt')
    
    return model

# %%
def train_phase2(model, train_loader, val_loader, epochs=3):
    # Apply LoRA to BERT
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none"
    )
    model.bert = get_peft_model(model.bert, lora_config)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Phase2 Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                anchor_out_ids=batch['anchor_out_ids'],
                anchor_out_mask=batch['anchor_out_mask'],  # Added
                anchor_in_ids=batch['anchor_in_ids'],
                anchor_in_mask=batch['anchor_in_mask'],    # Added
                domains_out=batch['domains_out'],
                domains_in=batch['domains_in'],
                numerics=batch['numerics']
            )
            
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    anchor_out_ids=batch['anchor_out_ids'],
                    anchor_out_mask=batch['anchor_out_mask'],  # Added
                    anchor_in_ids=batch['anchor_in_ids'],
                    anchor_in_mask=batch['anchor_in_mask'],    # Added
                    domains_out=batch['domains_out'],
                    domains_in=batch['domains_in'],
                    numerics=batch['numerics']
                )
                val_loss += criterion(outputs, batch['label']).item()
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    
    return model

# %%
from torch.utils.data import DataLoader
train_df, val_df, mlb, scaler = preprocess_dataframe(df)

train_dataset = WebPageDataset(train_df, bert_tokenizer, mlb)
val_dataset = WebPageDataset(val_df, bert_tokenizer, mlb)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# %%
model=MultiModalWebClassifier(domain_dim=mlb.classes_.shape[0]).to(device)

# %%


# %%

phase1_model = train_phase1(model, train_loader, val_loader)

# %%
   
    # Phase 2: LoRA fine-tuning
phase2_model = train_phase2(phase1_model, train_loader, val_loader)
    
    # Save final model
torch.save(phase2_model.state_dict(), 'final_model.pt')

# %%

import torch
torch.cuda.empty_cache()




# %%
