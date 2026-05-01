import torch
import torch.nn as nn
import numpy as np

class TransformerPlanner(nn.Module):
    """
    A Transformer-based strategic planner.
    Takes a sequence of previous states and actions and predicts the next optimal macro/goal.
    """
    def __init__(self, obs_dim=144, action_dim=18, d_model=64, n_heads=4, num_layers=2, num_macros=10):
        super(TransformerPlanner, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.state_emb = nn.Linear(obs_dim, d_model)
        self.action_emb = nn.Embedding(action_dim, d_model)
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 100, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer for selecting a macro or goal
        self.macro_head = nn.Linear(d_model, num_macros)

    def forward(self, states, actions):
        """
        states: (batch_size, seq_len, obs_dim)
        actions: (batch_size, seq_len)
        """
        seq_len = states.size(1)
        
        # Project state and embed actions
        s_emb = self.state_emb(states)
        a_emb = self.action_emb(actions)
        
        # We can alternate or sum them. For simplicity, let's sum them.
        x = s_emb + a_emb + self.pos_emb[:, :seq_len, :]
        
        # Causal mask to prevent looking into the future
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(states.device)
        
        out = self.transformer(x, mask=mask)
        
        # Predict next macro for the last timestep
        logits = self.macro_head(out[:, -1, :])
        return logits

    def select_macro(self, state_seq, action_seq, device='cpu'):
        self.eval()
        with torch.no_grad():
            s = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(action_seq, dtype=torch.long).unsqueeze(0).to(device)
            
            logits = self(s, a)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
        return action
