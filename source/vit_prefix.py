import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import timm

class vitPrefix(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=1000, n=4):
        super(vitPrefix, self).__init__()
        self.n = n
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.embedding_dim = self.model.embed_dim
        
        self.pretrained_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').register_tokens

    def forward(self, x, token_type='random', index=100, device='cuda', dataset=None):
        B = x.shape[0]  # Batch size
                
        # Get the patch embeddings from the model's patch embedding layer
        x = self.model.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.model.pos_embed[:, 1:x.shape[1]+1, :]
        
        # Generate tokens based on token_type
        if token_type == 'random':
            tokens = torch.randn(B, self.n, self.embedding_dim, device=x.device)
        elif token_type == 'ones':
            tokens = torch.ones(B, self.n, self.embedding_dim, device=x.device)
        elif token_type == 'zeros':
            tokens = torch.zeros(B, self.n, self.embedding_dim, device=x.device)
        elif token_type == 'repeat':
            tokens = x[:, 0:1, :].repeat(1, self.n, 1)
        elif token_type == 'random_patch':
            tokens = self.embed_random_image(index, device, dataset).repeat(1, self.n, 1)
        elif token_type == 'pretrained_reg' and self.n == 4:
            tokens = self.pretrained_reg.repeat(B, 1, 1)
        
        # Prepend the tokens to the sequence
        x = torch.cat((tokens, x), dim=1)
        
        # Add the class token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Apply the transformer layers
        x = self.model.blocks(x)
        
        # Layer norm
        x = self.model.norm(x)
        
        # Classification head
        x = self.model.head(x[:, 0])
        
        return x
    
    def embed_random_image(self, index, device, dataset):             
        # Select an image at a specific index from the dataset
        image, _ = dataset[index]
        
        # Add a batch dimension to the image
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = self.model.patch_embed(image)
        
        # Extract the nth token
        token = output[:, 0, :]
        
        return token

# # Example usage:
# model = vitPrefix(model_name='vit_base_patch16_224', pretrained=True, num_classes=1000, n=3)

# # Example inference with different token types
# input_tensor = torch.randn(1, 3, 224, 224)  # Dummy input tensor

# # Inference with random tokens
# output_random = model(input_tensor, token_type='random')
# print("Output with random tokens:", output_random)

# # Inference with ones tokens
# output_ones = model(input_tensor, token_type='ones')
# print("Output with ones tokens:", output_ones)

# # Inference with zeros tokens
# output_zeros = model(input_tensor, token_type='zeros')
# print("Output with zeros tokens:", output_zeros)
