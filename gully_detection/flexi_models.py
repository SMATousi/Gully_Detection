from flexivit_pytorch import (flexivit_base, flexivit_huge, flexivit_large,
                              flexivit_small, flexivit_tiny)
import torch
import numpy as np
import re

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
net = flexivit_small()

# --- Load .npz checkpoint ---
ckpt_path = "/root/flexi_models/flexivit_s_i1k.npz"
print(f"Loading checkpoint from {ckpt_path}...")

try:
    ckpt = np.load(ckpt_path, allow_pickle=True)
    print("Successfully loaded checkpoint.")
    
    # Print sample of the checkpoint keys for debugging
    print(f"Found {len(ckpt.files)} keys in checkpoint")
    if len(ckpt.files) > 5:
        print("Sample keys:", ckpt.files[:5], "...")
    else:
        print("All keys:", ckpt.files)
    
    # Prepare state_dict mappings
    state_dict = net.state_dict()
    print(f"Model has {len(state_dict)} parameters")
    
    # Debug: Print some model keys to compare with checkpoint keys
    model_keys = list(state_dict.keys())
    print("Model keys (sample):", model_keys[:5], "...")
    
    # Key mapping function (convert checkpoint keys to model keys)
    def map_key(ckpt_key):
        # Strip prefix if any (common in .npz files)
        # Example: 'model/encoder/layer1/conv' -> 'encoder.layer1.conv'
        clean_key = ckpt_key.replace('model/', '')
        # Replace '/' with '.' for PyTorch convention
        clean_key = clean_key.replace('/', '.')
        # Handle other common patterns
        clean_key = clean_key.replace('kernel', 'weight')
        return clean_key
    
    # Try different key mapping approaches
    new_state_dict = {}
    success_count = 0
    direct_match_found = False
    
    # First check if direct matching works
    direct_matches = sum(1 for k in state_dict.keys() if k in ckpt)
    if direct_matches > 0:
        print(f"Found {direct_matches} direct key matches, using direct mapping")
        direct_match_found = True
        for k in state_dict.keys():
            if k in ckpt:
                # Convert numpy array to tensor
                new_state_dict[k] = torch.from_numpy(ckpt[k])
                success_count += 1
            else:
                print(f"Warning: {k} not found in checkpoint")
    
    # If direct matching failed, try mapping
    if not direct_match_found:
        print("No direct matches found, trying key mapping...")
        # Try to map each checkpoint key to a model key
        ckpt_to_model_map = {}
        for ckpt_key in ckpt.files:
            for model_key in state_dict.keys():
                # Check if the end of the key matches (often most reliable)
                if model_key.split('.')[-1] == ckpt_key.split('/')[-1] or \
                   model_key.endswith(ckpt_key.split('/')[-1]):
                    ckpt_to_model_map[ckpt_key] = model_key
                    break
        
        print(f"Found {len(ckpt_to_model_map)} potential key mappings")
        
        # Load weights using the mapping
        for ckpt_key, model_key in ckpt_to_model_map.items():
            try:
                param = torch.from_numpy(ckpt[ckpt_key])
                # Check if shapes match, transpose if needed (common for conv weights)
                if param.shape != state_dict[model_key].shape:
                    # For conv weights, may need to transpose
                    if len(param.shape) == 4 and len(state_dict[model_key].shape) == 4:
                        param = param.permute(3, 2, 0, 1)  # Common transpose pattern
                    # For linear weights
                    elif len(param.shape) == 2 and len(state_dict[model_key].shape) == 2:
                        param = param.transpose(0, 1)
                    
                    # Check again
                    if param.shape != state_dict[model_key].shape:
                        print(f"Shape mismatch for {model_key}: {param.shape} vs {state_dict[model_key].shape}")
                        continue
                
                new_state_dict[model_key] = param
                success_count += 1
            except Exception as e:
                print(f"Error processing {ckpt_key} -> {model_key}: {e}")
    
    # Load weights into model
    if success_count > 0:
        print(f"Successfully mapped {success_count}/{len(state_dict)} parameters")
        missing, unexpected = net.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        if len(missing) > 0 and len(missing) <= 10:
            print("  ", missing)
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) > 0 and len(unexpected) <= 10:
            print("  ", unexpected)
    else:
        print("Failed to map any parameters. The checkpoint format may be incompatible.")
    
    # Move model to device
    net = net.to(device)
    print("Model loaded and moved to device.")
    
    # Test with a sample input
    net.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 128, 128).to(device)
        print("Input shape:", test_input.shape)
        output = net(test_input)
        print("Output shape:", output.shape)
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")


# print(net(img))