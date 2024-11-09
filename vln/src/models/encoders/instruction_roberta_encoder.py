from .bert_backbone import *
from .lora import LinearWithLoRA
from functools import partial

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_text_encoder

        self.embeddings = RobertaEmbeddings(config)
                
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False
        
        if config.LORA.add_for_instruction_encoder:
            assign_lora = partial(LinearWithLoRA, rank=config.LORA.lora_r, alpha=config.LORA.lora_alpha)
            lora_start_layer = config.LORA.lora_start_layer if config.LORA.lora_start_layer<len(self.layer) else 0
            for layer_idx, layer in enumerate(self.layer):
                if layer_idx >= lora_start_layer:
                    if config.LORA.lora_query:
                        layer.attention.self.query = assign_lora(layer.attention.self.query)
                    if config.LORA.lora_key:
                        layer.attention.self.key = assign_lora(layer.attention.self.key)
                    if config.LORA.lora_value:
                        layer.attention.self.value = assign_lora(layer.attention.self.value)
                    if config.LORA.lora_projection:
                        layer.attention.output.dense = assign_lora(layer.attention.out.dense)
                    if config.LORA.lora_mlp:
                        layer.intermediate.dense = assign_lora(layer.intermediate.dense)
                        layer.output.dense = assign_lora(layer.output.dense)
            
            print("Add LoRA for instruction encoder")

    def forward(self, txt_inputs, txt_masks=None):
        txt_inputs = txt_inputs.long()
        if txt_masks is None:
            txt_masks = (txt_inputs != 1).to(txt_inputs.device)
        txt_embeds = self.embeddings(txt_inputs)
        
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        
        return txt_embeds, txt_masks, txt_embeds[:,0,:]