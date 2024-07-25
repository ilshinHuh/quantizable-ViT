import torch
import timm
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os

from vit_prefix import vitPrefix

# Check if CUDA is available and use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained Vision Transformer model from timm and modify the head for 1000 classes (ImageNet)
def_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
def_model.to(device)
def_model.eval()

prefix_num = 4
pre_model = vitPrefix(n = prefix_num)
pre_model.to(device)
pre_model.eval()

rep_model = vitPrefix(n = prefix_num)
rep_model.to(device)
rep_model.eval()

ran_model = vitPrefix(n = prefix_num)
ran_model.to(device)
ran_model.eval()

# Define a hook to capture activations
activation_dict_def_model = {}
activation_dict_pre_model = {}
activation_dict_rep_model = {}
activation_dict_ran_model = {}

def hook_fn_def_model(module, input, output):
    activation_dict_def_model[module] = output

def hook_fn_pre_model(module, input, output):
    activation_dict_pre_model[module] = output
    
def hook_fn_rep_model(module, input, output):
    activation_dict_rep_model[module] = output
    
def hook_fn_ran_model(module, input, output):
    activation_dict_ran_model[module] = output

# Register hooks to transformer block outputs only for def_model
hooks_def_model = []
for name, module in def_model.named_modules():
    if isinstance(module, timm.models.vision_transformer.Block):
        hook = module.register_forward_hook(hook_fn_def_model)
        hooks_def_model.append(hook)

# Register hooks to transformer block outputs only for pre_model
hooks_pre_model = []
for name, module in pre_model.named_modules():
    if isinstance(module, timm.models.vision_transformer.Block):
        hook = module.register_forward_hook(hook_fn_pre_model)
        hooks_pre_model.append(hook)
        
# Register hooks to transformer block outputs only for rep_model
hooks_rep_model = []
for name, module in rep_model.named_modules():
    if isinstance(module, timm.models.vision_transformer.Block):
        hook = module.register_forward_hook(hook_fn_rep_model)
        hooks_rep_model.append(hook)
        
# Register hooks to transformer block outputs only for ran_model
hooks_ran_model = []
for name, module in ran_model.named_modules():
    if isinstance(module, timm.models.vision_transformer.Block):
        hook = module.register_forward_hook(hook_fn_ran_model)
        hooks_ran_model.append(hook)

# Preprocessing for ImageNet dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet dataset
# Assuming ImageNet data is available in the 'path_to_imagenet' directory
imagenet_data_dir = '/data/ILSVRC2012' 
dataset = datasets.ImageFolder(root=os.path.join(imagenet_data_dir, 'val'), transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create a directory to save results
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# Get a single image and label from the dataset
fixed_index = 33  # Choose a fixed index for consistency
input_image, label = dataset[fixed_index]
input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

# Perform a forward pass to capture activations
with torch.no_grad():
    output = def_model(input_image)
    output = pre_model(input_image, token_type='pretrained_reg')
    output = rep_model(input_image, token_type='repeat')
    output = ran_model(input_image, token_type='random_patch', device=device, dataset=dataset)

# Unregister hooks for def_model
for hook in hooks_def_model:
    hook.remove()

# Unregister hooks for pre_model
for hook in hooks_pre_model:
    hook.remove()
    
# Unregister hooks for rep_model
for hook in hooks_rep_model:
    hook.remove()
    
# Unregister hooks for ran_model
for hook in hooks_ran_model:
    hook.remove()

# # Save input image
# input_image_path = os.path.join(output_dir, f'input_image_{prefix_num}.png')
# input_image_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
# input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())  # Normalize to [0, 1]
# plt.imsave(input_image_path, input_image_np)

# # Save classification result
# classes = dataset.classes
# predicted_class = classes[output.argmax().item()]
# classification_result_path = os.path.join(output_dir, f'classification_result_{prefix_num}_pat.txt')
# with open(classification_result_path, 'w') as f:
#     f.write(f'Predicted class: {predicted_class}\n')
#     f.write(f'True label: {classes[label]}\n')
    
# Plot and save activations
def plot_activations(activation_dict, output_dir):
    num_blocks = len(activation_dict)

    fig, axs = plt.subplots(1, num_blocks, figsize=(18 * num_blocks, 15))  # Adjust the figure size
    if num_blocks == 1:
        axs = [axs]
    for i, (layer, activation) in enumerate(activation_dict.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            activation = torch.abs(activation).permute(0, 2, 1).cpu().numpy()  # Rearrange dimensions to (batch size, embedding dim, sequence length)
            im = axs[i].imshow(activation[0], aspect='auto', cmap='viridis', interpolation='none')
            fig.colorbar(im, ax=axs[i])
            axs[i].set_xlabel('Sequence Length')
            axs[i].set_ylabel('Embedding Dimension')
            axs[i].set_title(f'Block: {i}')
    
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'merged_activations_repeat_{prefix_num}.png'))
    plt.savefig(os.path.join(output_dir, f'merged_activations_default.png'))
    plt.close()

def plot_token_norms(activation_dict_def_model, activation_dict_pre_model, activation_dict_rep_model, activation_dict_ran_model, output_dir):
    num_blocks = len(activation_dict_def_model)

    fig, axs = plt.subplots(1, num_blocks, figsize=(8 * num_blocks, 8))  # Adjust the figure size
    if num_blocks == 1:
        axs = [axs]
    for i, (layer, activation) in enumerate(activation_dict_def_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.norm(activation, dim=2).cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            token_norms = torch.cat((torch.full((4,), float('nan')), token_norms[:-4]), dim=0)  # Shift the norms to the right by four indices
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='red', linewidth=0.5, markersize=2, marker='o', label='def_model')  # Set markersize to a smaller value and add marker='o'
            axs[i].set_xlabel('Token Index')
            axs[i].set_ylabel('Token Norm')
            axs[i].set_title(f'Block: {i}')
            
    for i, (layer, activation) in enumerate(activation_dict_pre_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.norm(activation, dim=2).cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='blue', linewidth=0.5, markersize=2, marker='o', label='pre_model')  # Set markersize to a smaller value and add marker='o'
            
    for i, (layer, activation) in enumerate(activation_dict_rep_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.norm(activation, dim=2).cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='green', linewidth=0.5, markersize=2, marker='o', label='rep_model')  # Set markersize to a smaller value and add marker='o'
            
    for i, (layer, activation) in enumerate(activation_dict_ran_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.norm(activation, dim=2).cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='orange', linewidth=0.5, markersize=2, marker='o', label='ran_model')  # Set markersize to a smaller value and add marker='o'
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'token_norms_33.png'), dpi=300)
    plt.close()
    
def plot_token_max(activation_dict_def_model, activation_dict_pre_model, activation_dict_rep_model, activation_dict_ran_model, output_dir):
    num_blocks = len(activation_dict_def_model)

    fig, axs = plt.subplots(1, num_blocks, figsize=(8 * num_blocks, 8))  # Adjust the figure size
    if num_blocks == 1:
        axs = [axs]
    for i, (layer, activation) in enumerate(activation_dict_def_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_max = torch.max(activation, dim=2).values.cpu()  # Calculate the max value of tokens along the embedding dimension
            token_max = token_max.squeeze()  # Remove the batch dimension
            token_max = torch.cat((torch.full((4,), float('nan')), token_max[:-4]), dim=0)  # Shift the max values to the right by four indices
            num_tokens = token_max.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_max, color='red', linewidth=0.5, markersize=2, marker='o', label='def_model')  # Set markersize to a smaller value and add marker='o'
            axs[i].set_xlabel('Token Index')
            axs[i].set_ylabel('Token Max Value')
            axs[i].set_title(f'Block: {i}')
            
    for i, (layer, activation) in enumerate(activation_dict_pre_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_max = torch.max(activation, dim=2).values.cpu()  # Calculate the max value of tokens along the embedding dimension
            token_max = token_max.squeeze()  # Remove the batch dimension
            num_tokens = token_max.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_max, color='blue', linewidth=0.5, markersize=2, marker='o', label='pre_model')  # Set markersize to a smaller value and add marker='o'
            
    for i, (layer, activation) in enumerate(activation_dict_rep_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.max(activation, dim=2).values.cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='green', linewidth=0.5, markersize=2, marker='o', label='rep_model')  # Set markersize to a smaller value and add marker='o'
            
    for i, (layer, activation) in enumerate(activation_dict_ran_model.items()):
        if activation.dim() == 3:  # Check if activation is a 3D tensor (batch size, sequence length, embedding dim)
            token_norms = torch.max(activation, dim=2).values.cpu()  # Calculate the norm of tokens along the embedding dimension
            token_norms = token_norms.squeeze()  # Remove the batch dimension
            num_tokens = token_norms.shape[0]
            token_indices = torch.arange(num_tokens)
            axs[i].plot(token_indices, token_norms, color='orange', linewidth=0.5, markersize=2, marker='o', label='ran_model')  # Set markersize to a smaller value and add marker='o'
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'token_max_33.png'), dpi=300)
    plt.close()
    
plot_token_norms(activation_dict_def_model, activation_dict_pre_model, activation_dict_rep_model, activation_dict_ran_model, output_dir)
plot_token_max(activation_dict_def_model, activation_dict_pre_model, activation_dict_rep_model, activation_dict_ran_model, output_dir)
# plot_activations(activation_dict, output_dir)