import torch
import timm
from torchsummary import summary
from vit_prefix import vitPrefix

# Load a pretrained Vision Transformer model from timm
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
# model = vitPrefix()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the model summary
# The input size should match the expected input size for the model (3, 224, 224) for ViT
# summary(model, input_size=(3, 224, 224))

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
print(model.register_tokens.shape)