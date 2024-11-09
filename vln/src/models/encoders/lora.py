import torch
import torch.nn as nn

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

class MultiheadAttnWithLoRA(torch.nn.Module):
    def __init__(self, attn, rank, alpha):
        super().__init__()
        self.attn = attn
        self.lora = LoRALayer(attn.out_proj.in_features, attn.out_proj.out_features, rank, alpha)
    
    def forward(self, x):
        return self.attn(x) + self.lora(x)