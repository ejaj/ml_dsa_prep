import torch 
import torch.nn as nn 

def check_dead_neurons(model, x):
    activations = []
    def hook_fn(_, _, output):
        activations.append(output.detach())
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(hook_fn))
    # Forward pass
    with torch.no_grad():
        model(x)
    # Analyze activations
    for i, act in enumerate(activations):
        zero_ratio = (act == 0).float().mean.item()
        print(f"Layer {i+1}: {zero_ratio*100:.2f}% neurons dead")

     # Remove hooks
    for h in hooks:
        h.remove()


 # Check gradient norms for dying neurons
# for name, param in model.named_parameters():
#     if "weight" in name:
#         grad_norm = param.grad.abs().mean().item()
#         print(f"{name} gradient mean: {grad_norm:.6f}")
