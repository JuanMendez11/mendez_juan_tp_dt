import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class DecisionTransformerHF(nn.Module):
    """
    Decision Transformer usando GPT-2 de HuggingFace como backbone.
    
    Ventajas:
    - Código más simple (no implementar atención desde cero)
    - Usa componentes probados y optimizados
    - Fácil de modificar y experimentar
    
    Arquitectura:
    1. Embeddings: group, state (rating), action (item), rtg
    2. GPT-2 Transformer (de HuggingFace)
    3. Action head: predice siguiente item
    """
    
    def __init__(
        self,
        num_items=752,
        num_groups=8,
        hidden_dim=128,
        n_layers=3,
        n_heads=4,
        max_seq_len=50,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_items = num_items
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # === EMBEDDINGS ===
        
        # Group embedding: representa el cluster del usuario (0-7)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        
        # Action embedding: representa el item (película/libro)
        # Embedding de alta dimensión para capturar similitud entre items
        self.action_embedding = nn.Embedding(num_items, hidden_dim)
        
        # State embedding: convierte rating (1-5) a vector
        # Usamos linear en vez de embedding porque ratings pueden ser continuos
        self.state_embedding = nn.Linear(1, hidden_dim)
        
        # RTG embedding: convierte return-to-go a vector
        self.rtg_embedding = nn.Linear(1, hidden_dim)
        
        # Timestep embedding: codifica posición temporal
        self.timestep_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # === GPT-2 TRANSFORMER (HuggingFace) ===
        
        # Configuración del GPT-2
        config = GPT2Config(
            vocab_size=1,  # No usamos vocabulario de palabras
            n_positions=max_seq_len * 4,  # 4 tokens por paso (group, state, action, rtg)
            n_embd=hidden_dim,
            n_layer=n_layers,
            n_head=n_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        
        # Transformer de HuggingFace (¡esto reemplaza ~200 líneas de código!)
        self.transformer = GPT2Model(config)
        
        # === OUTPUT HEAD ===
        
        # Layer normalization final
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        # Action prediction head: hidden_dim → num_items (scores)
        self.action_head = nn.Linear(hidden_dim, num_items)
        
        # Inicialización de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Inicialización de pesos (como en GPT-2)"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, states, actions, rtgs, timesteps, groups, attention_mask=None):
        """
        Forward pass del Decision Transformer.
        
        Args:
            states: (batch, seq_len, 1) - ratings de items vistos
            actions: (batch, seq_len) - IDs de items vistos
            rtgs: (batch, seq_len, 1) - return-to-go en cada paso
            timesteps: (batch, seq_len) - timestep de cada posición
            groups: (batch,) - grupo del usuario
            attention_mask: (batch, seq_len) - máscara de padding (opcional)
        
        Returns:
            action_logits: (batch, seq_len, num_items) - scores para predecir siguiente item
        """
        batch_size, seq_len = actions.shape
        
        # === CREAR EMBEDDINGS ===
        
        # Group embedding: expandir para toda la secuencia
        # (batch,) → (batch, 1, hidden_dim) → (batch, seq_len, hidden_dim)
        group_emb = self.group_embedding(groups).unsqueeze(1)
        group_emb = group_emb.expand(batch_size, seq_len, self.hidden_dim)
        
        # State embedding: ratings
        state_emb = self.state_embedding(states)  # (batch, seq_len, hidden_dim)
        
        # Action embedding: items
        action_emb = self.action_embedding(actions)  # (batch, seq_len, hidden_dim)
        
        # RTG embedding: return-to-go
        rtg_emb = self.rtg_embedding(rtgs)  # (batch, seq_len, hidden_dim)
        
        # Timestep embedding: posición temporal
        timestep_emb = self.timestep_embedding(timesteps)  # (batch, seq_len, hidden_dim)
        
        # === INTERLEAVE: [group, state, action, rtg] para cada timestep ===
        
        # Crear secuencia alternada: group_0, state_0, action_0, rtg_0, group_1, ...
        # Shape final: (batch, seq_len * 4, hidden_dim)
        stacked_inputs = torch.stack(
            [group_emb, state_emb, action_emb, rtg_emb], dim=2
        ).reshape(batch_size, seq_len * 4, self.hidden_dim)
        
        # Agregar timestep embedding (se repite 4 veces por paso)
        # (batch, seq_len, hidden_dim) → (batch, seq_len * 4, hidden_dim)
        timestep_emb_expanded = timestep_emb.repeat_interleave(4, dim=1)
        stacked_inputs = stacked_inputs + timestep_emb_expanded
        
        # === CREAR ATTENTION MASK ===
        
        if attention_mask is not None:
            # Expandir máscara para los 4 tokens por paso
            # (batch, seq_len) → (batch, seq_len * 4)
            attention_mask = attention_mask.repeat_interleave(4, dim=1)
        
        # === PASAR POR TRANSFORMER (HuggingFace hace la magia) ===
        
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask
        )
        
        # Obtener hidden states: (batch, seq_len * 4, hidden_dim)
        hidden_states = transformer_outputs.last_hidden_state
        
        # === EXTRACT ACTION PREDICTIONS ===
        
        # Queremos predecir acciones, así que tomamos los hidden states
        # en las posiciones de "action" (cada 4 tokens, offset 2)
        # Posiciones: 2, 6, 10, 14, ... (group=0, state=1, action=2, rtg=3)
        action_hidden = hidden_states[:, 2::4, :]  # (batch, seq_len, hidden_dim)
        
        # Layer norm
        action_hidden = self.ln_f(action_hidden)
        
        # Predecir siguiente item
        action_logits = self.action_head(action_hidden)  # (batch, seq_len, num_items)
        
        return action_logits