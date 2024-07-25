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
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
prefix_num = 16
model = vitPrefix(n = prefix_num)
model.to(device)
model.eval()

# Define a hook to capture activations
activation_dict = {}

def hook_fn(module, input, output):
    activation_dict[module] = output

# Register hooks to transformer block outputs only
hooks = []
for name, module in model.named_modules():
    if isinstance(module, timm.models.vision_transformer.Block):
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

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
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get a single image and label from the dataset
fixed_index = 0  # Choose a fixed index for consistency
input_image, label = dataset[fixed_index]
input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

# Perform a forward pass to capture activations
with torch.no_grad():
    output = model(input_image, token_type='random_patch', index=10, device=device, dataset=dataset)
    # output = model(input_image)

# Unregister hooks
for hook in hooks:
    hook.remove()

# Save input image
input_image_path = os.path.join(output_dir, f'input_image_{prefix_num}.png')
input_image_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())  # Normalize to [0, 1]
plt.imsave(input_image_path, input_image_np)

# Save classification result
classes = dataset.classes
predicted_class = classes[output.argmax().item()]
classification_result_path = os.path.join(output_dir, f'classification_result_{prefix_num}_pat.txt')
with open(classification_result_path, 'w') as f:
    f.write(f'Predicted class: {predicted_class}\n')
    f.write(f'True label: {classes[label]}\n')
    
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
    plt.savefig(os.path.join(output_dir, f'merged_activations_{prefix_num}_pat.png'))
    plt.close()

plot_activations(activation_dict, output_dir)