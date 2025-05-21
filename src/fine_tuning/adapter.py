import torch
from torch import nn

from src.models.basemodel import Model


class Adapter(nn.Module):
    def __init__(self, original_layer: nn.Module,
                 size: int, model_dim: int) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.adapter_block = nn.Sequential(
            nn.Linear(model_dim, size),
            nn.ReLU(),
            nn.Linear(size, model_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.original_layer(x)
        return (self.adapter_block(output) + output)


def add_adapters_to_model_layers(model: Model, layers: int, proj_size: int) -> None:
    layernorm_layers = ["layernorm", "ln_1", "ln_2"]
    
    for layer_name, param in model.model.named_parameters():
        if not any(name in layer_name for name in layernorm_layers):
            param.requires_grad = False
        
    for i in range(11, 11-layers, -1):
        model.model.decoder.transformer.h[i].attn.c_proj = \
            Adapter(model.model.decoder.transformer.h[i].attn.c_proj, proj_size, 768)
        model.model.decoder.transformer.h[i].mlp.c_proj = \
            Adapter(model.model.decoder.transformer.h[i].mlp.c_proj, proj_size, 768)
        model.model.decoder.transformer.h[i].crossattention.c_proj = \
            Adapter(model.model.decoder.transformer.h[i].crossattention.c_proj, proj_size, 768)
            
        model.model.encoder.encoder.layer[i].output.dense = \
            Adapter(model.model.encoder.encoder.layer[i].output.dense, proj_size, 768)
        model.model.encoder.encoder.layer[i].attention.output.dense = \
            Adapter(model.model.encoder.encoder.layer[i].attention.output.dense, proj_size, 768)
    
    model.to_cuda()
