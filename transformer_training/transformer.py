import torch
#Imports the main PyTorch package used for tensor operations and building neural networks.
from torch import nn
#Imports the nn module, which contains classes to help build neural network layers.
from torch.nn.modules.linear import Linear
#Linear: a fully connected layer.
from torch.nn.modules.dropout import Dropout
#Dropout: a dropout layer for regularization.
from torch.nn.modules.normalization import LayerNorm
#LayerNorm: a normalization layer applied across the features of a layer.
import math
import pandas as pd
from typing import Sequence, Tuple

from torch.utils.data import DataLoader, random_split
#DataLoader: for batching and shuffling data.
#random_split: for splitting datasets into training/validation parts.

# ===========================
# 1. Define Transformer Components with Complexity
# ===========================

class PositionalEncoding(nn.Module):
# Defines a new module (subclass of nn.Module) to add positional information to input features.
    def __init__(self, d_model, max_len=100):
    #The constructor takes:d_model: the dimensionality of the input features.
    #max_len: the maximum length of the input sequence (default is 100).
        super(PositionalEncoding, self).__init__()
        #Calls the constructor of the parent nn.Module class
        pe = torch.zeros(max_len, d_model)
        #Creates a tensor pe of zeros with shape [max_len, d_model] to hold positional encodings.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #Generates a tensor of positions from 0 to max_len-1 and reshapes it to a column vector 
        #of (shape [max_len, 1]).
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #Creates a scaling factor for each even-indexed dimension. 
        #This uses an exponential decay factor so that the frequencies of sine and cosine vary with 
        #the dimension index.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        #Registers pe as a buffer. 
        #This means it’s not a learnable parameter but will be saved as part of the model’s state.

    def forward(self, x):
        # Defines the forward pass. 
        # The input x is expected to have shape [N, T, d_model] where N is batch size, 
        # T is sequence length.
        # x shape: [N, T, d_model]
        return x + self.pe[:x.shape[1], :]
        # Adds the pre-computed positional encodings to the input features. 
        # The encoding is sliced to match the sequence length T
    

class Gate(nn.Module):
# Defines a custom gating mechanism as a neural network module.
    def __init__(self, d_input, d_output, beta=0.9, momentum=0.9):
        #d_input: dimension of the input features to the gate.
        #d_output: dimension used to scale the output.
        #beta: a temperature parameter to control the softmax sharpness.
        #momentum: a momentum term for updating running weights.
        super().__init__()
        #Calls the parent class constructor.
        self.trans = nn.Linear(d_input, d_output)
        # Defines a linear transformation mapping the input to a new space of size d_output
        self.d_output = d_output
        self.t = beta
        self.momentum = momentum
        # Saves the output dimension, beta (renamed as t), and momentum as instance variables
        self.register_buffer('dynamic_weights', None, persistent=False)
        # Register a buffer to hold the dynamic gating weights.

    def forward(self, gate_input):
        # gate_input shape: [N, F']
        current_gate = torch.softmax(self.trans(gate_input) / self.t, dim=-1)
        # If no dynamic weights have been stored, initialize them.
        if self.dynamic_weights is None:
            self.dynamic_weights = current_gate.detach()
        else:
            # Ensure correct shape before updating
            if self.dynamic_weights.shape != current_gate.shape:
                print(f"Shape Mismatch! dynamic_weights: {self.dynamic_weights.shape}, current_gate: {current_gate.shape}")
                self.dynamic_weights = torch.zeros_like(current_gate).to(current_gate.device)

            # Update dynamic weights using momentum (detached from gradient)
            self.dynamic_weights = self.momentum * self.dynamic_weights + (1 - self.momentum) * current_gate.detach()
            #First call: If no dynamic weights exist, initialize them with the current gate 
            # (detached from the gradient).
            #Subsequent calls: Update the running dynamic weights using an exponential 
            # moving average controlled by momentum
        
        alpha = 0.8  # More weight on the current gate.
        combined_gate = alpha * current_gate + (1 - alpha) * self.dynamic_weights
        # Averages the current gate with the dynamic (running) gate to obtain a combined gating signal.
        
        output = self.d_output * combined_gate
        
        # Reset dynamic_weights at the end of the forward pass (i.e., after processing the current batch)
        self.dynamic_weights = None
        
        return output
        # Scales the combined gate by d_output and returns it. 
        # This output will later be used to weight the input features.

class TAttention(nn.Module):
# Defines a custom multi-head attention module for temporal (intra-stock) aggregation.   
    def __init__(self, d_model, nhead, dropout):
        # d_model: model (feature) dimension.
        # nhead: number of attention heads.
        # dropout: dropout rate applied to attention weights and feed-forward layers.
        super().__init__()
        # Calls the parent class constructor.
        self.d_model = d_model
        self.nhead = nhead
        # Stores the model dimension and number of heads.
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        # Defines linear layers to project input features into queries, keys, and values. 
        # Note that no bias is used.

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)
        # Sets up a dropout layer for each head if dropout is specified. 
        # The dropout layers are stored in a ModuleList for proper registration.
        
        # Input normalization layer
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # Post-attention normalization layer for FFN
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )
        # Builds a feed-forward network (FFN) as a sequence of layers:
        # A linear layer followed by ReLU activation.
        # A dropout layer.
        # Another linear layer followed by dropout.
        # This FFN further processes the output after attention.
        
    def forward(self, x):
        # x shape: [N, T, d_model]
        x = self.norm1(x)
        # Normalizes the input using the first normalization layer.
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)
        # Projects the normalized input into query, key, and value representations.

        dim = int(self.d_model / self.nhead)
        att_output = []
        # Calculates the dimension for each attention head and prepares a list to hold outputs from each head.
        for i in range(self.nhead):
        # Iterates over each head.
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            # Splits the query, key, and value tensors into head-specific chunks. 
            # The last head takes the remaining dimensions to avoid slicing issues.
            
            scaling_factor = math.sqrt(qh.size(-1))
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / scaling_factor, dim=-1)
            # Computes the attention weight matrix for the current head:
            # Multiplies queries with the transpose of keys.
            # Applies softmax to get normalized attention scores.
            
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            # Applies dropout to the attention weights for the current head if dropout layers are defined.
            
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
            # Uses the attention weights to compute a weighted sum of the value vectors for the current 
            # head and appends the result.
        
        att_output = torch.concat(att_output, dim=-1)
        # # Concatenate all head outputs.
        
        xt = x + att_output
        # Adds a residual connection by summing the concatenated attention output with the original input.
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output
        # Returns the final output from the temporal attention module.
        
class SAttention(nn.Module):
# Defines a spatial attention module for inter-stock (or inter-feature) aggregation.
    def __init__(self, d_model, nhead, dropout):
        # Constructor parameters are similar to TAttention: model dimension, number of heads, and dropout rate.
        super().__init__()
        # Calls the parent class constructor.
        self.d_model = d_model
        self.nhead = nhead
        # Stores the model dimension and number of attention heads.
        self.temperature = math.sqrt(self.d_model / nhead)
        # Sets a temperature scaling factor using the square root of the head dimension. 
        # This is used to scale the dot-product in the softmax.

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        # Defines linear projections for queries, keys, and values (as in TAttention).
        
        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)
        # Creates and stores a dropout layer for each head in a ModuleList.
        
        # Input normalization
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN normalization
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # Defines two layer normalization modules: 
        # one before the attention and one after the residual connection.
        
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )
        # Constructs a feed-forward network identical in structure to the one in TAttention.

    def forward(self, x):
        # x shape: [N, T, d_model]
        x = self.norm1(x)
        # Transpose for head-wise operations: [T, N, d_model]
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        # Projects the normalized input into query, key, and value representations, then transposes them so 
        # that the time dimension becomes the first axis. This change facilitates head-wise operations.
                
        dim = int(self.d_model / self.nhead)
        att_output = []
        # Computes the size of each attention head and prepares an empty list for head outputs.
        
        for i in range(self.nhead):
            # Iterates over each attention head.
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            # Splits the transposed query, key, and value tensors into chunks corresponding to each head. 
            # The last head takes all remaining dimensions.
            
            # Compute scaled dot-product attention.
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            # Computes the scaled dot-product attention for the current head:
            # Multiplies the head-specific queries with the transposed keys.
            # Divides by the pre-computed temperature to stabilize gradients.
            # Applies softmax to obtain normalized attention scores.
            
            atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            # Applies the dropout layer for the current head to the attention weights.
            
            # Transpose back to original shape.
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
            # Multiplies the attention weights with the value vectors to get the head output, 
            # then transposes the result back to match the original dimensions before appending.
            
        att_output = torch.concat(att_output, dim=-1)
        # Concatenates the outputs from all attention heads along the feature dimension.
        
        xt = x + att_output
        # Adds a residual connection by summing the concatenated attention output with the original input.
        xt = self.norm2(xt)
        # Applies the second normalization.
        att_output = xt + self.ffn(xt)
        # Processes the normalized result through the feed-forward network and adds another residual connection.
        return att_output
        # Returns the final output from the spatial attention module.

class TemporalAttention(nn.Module):
# Defines a module to aggregate information across the time dimension.
    def __init__(self, d_model):
    # Constructor takes only the model dimension as input.
        super().__init__()
        # Calls the parent class constructor.
        self.trans = nn.Linear(d_model, d_model, bias=False)
        # Creates a linear layer that projects input features to the same dimension. 
        # This is used to compute an intermediate representation.

    def forward(self, z):
        # z shape: [N, T, d_model]
        h = self.trans(z)  
        # Project the features.
        # Applies the linear transformation to the entire sequence.
        
        query = h[:, -1, :].unsqueeze(-1)
        # Uses the last time step from the projected features as a query. 
        # The unsqueeze adds a singleton dimension to allow for matrix multiplication.
        
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T]
        # Computes a similarity score (dot product) between every time step in h and the query. 
        # Squeezing removes the extra dimension, resulting in a score for each time step.
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # [N, 1, T]
        # Applies softmax to obtain weights over the time steps and reshapes so that it can be 
        # used for weighted averaging.
        output = torch.matmul(lam, z).squeeze(1)  # [N, d_model]
        # Uses the computed weights to produce a weighted sum over the original input z, 
        # effectively aggregating the sequence into a single feature vector per batch element.
        return output
        # Returns the aggregated output.

# ===========================
# 2. Define MASTER Architecture without Final Prediction Head
# ===========================

class MASTER(nn.Module):
# Defines the overall model architecture named MASTER.
    def __init__(self,input_feature_ranges,gate_input_ranges,d_feat, d_model, t_nhead, s_nhead, 
                 T_dropout_rate, S_dropout_rate, beta, aggregate_output=False):
        """
        Args:
        The constructor takes several parameters:
        d_feat: the number of input features.
        d_model: the dimension after the first linear projection.
        t_nhead: number of attention heads for the temporal attention (TAttention).
        s_nhead: number of attention heads for the spatial attention (SAttention).
        T_dropout_rate and S_dropout_rate: dropout rates for the respective attention modules.
        gate_input_start_index and gate_input_end_index: indices that specify which features in the input 
        will be used for the gating mechanism.
        beta: parameter passed to the Gate (temperature for softmax scaling).
        aggregate_output: If True, aggregate daily embeddings into one vector per window.
                            If False, return sequential daily embeddings.
        """
        super(MASTER, self).__init__()
        # Calls the parent class constructor.
        
        # these two drive your slicing
        self.input_feature_ranges = input_feature_ranges
        self.gate_input_ranges    = gate_input_ranges
        self.aggregate_output = aggregate_output

        # Gate mechanism to dynamically weight features.
        # Stores the indices defining which features will serve as the gating input.
        
        # how many gate inputs?
        self.d_gate_input = sum((hi - lo) for lo, hi in gate_input_ranges)

        # the gate itself maps gate_inputs → a weight vector of length d_feat
        self.feature_gate = Gate(
            d_input  = self.d_gate_input,
            d_output = d_feat,
            beta     = beta,
            momentum = 0.9
        )
        
        # Compose all layers for feature transformation.
        self.layers = nn.Sequential(
            # Feature transformation layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # Intra-stock (temporal) aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # Inter-stock (spatial) aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            # Additional temporal aggregation
        )
        # Aggregator: aggregates T daily embeddings into one vector per window.
        self.aggregator = TemporalAttention(d_model)

    def forward(self, x, aggregate=None):
        """
        Args:
            x: Input tensor of shape [N, T, d_feat].
            aggregate (optional): Boolean flag. If True, returns aggregated embedding.
                                  If False, returns sequential daily embeddings.
                                  If None, uses the default (self.aggregate_output).
        Returns:
            If aggregate is True: Tensor of shape [N, d_model] (one vector per window).
            If aggregate is False: Tensor of shape [N, T, d_model] (one vector per day).
        """
        # x shape: [N, T, d_feat]
        # Extract the source features.
        # x shape: [N, T, d_feat]
        #print("Input x shape:", x.shape)  # Debugging the input shape
        
        N, T, _ = x.shape

        # 1) build src by concatenating each input_feature_range slice
        src_slices = [ x[:, :, s:e] for (s, e) in self.input_feature_ranges ]
        src = torch.cat(src_slices, dim=2)   # [N, T, d_feat]

        # 2) build gate_input by concatenating each gate_input_range at the last time step
        gate_slices = [ x[:, -1, s:e] for (s, e) in self.gate_input_ranges ]
        gate_input = torch.cat(gate_slices, dim=1)  # [N, d_gate_input]

        if gate_input.shape[-1] != self.d_gate_input:
            raise RuntimeError(f"Expected gate_input width {self.d_gate_input}, got {gate_input.shape[-1]}")

        gated_weights = self.feature_gate(gate_input)  # → [N, d_feat]
        src = src * gated_weights.unsqueeze(1)         # broadcast over time → [N, T, d_feat]

        # 3) feed through transformer trunk
        out = self.layers(src)  # [N, T, d_model]

        # 4) optional aggregation
        if aggregate is None:
            aggregate = self.aggregate_output
        if aggregate:
            out = self.aggregator(out)  # [N, d_model]

        return out