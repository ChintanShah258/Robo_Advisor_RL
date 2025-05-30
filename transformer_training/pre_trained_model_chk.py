# inspect_checkpoint.py
import torch, os

save_path   = 'robo_advisor/transformer_training/model'
save_prefix = 'sp500_master'
checkpoint_path = os.path.join(save_path, f"{save_prefix}_pretrained.pth")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

print("\n=== MASTER parameters ===")
for name, tensor in checkpoint['model'].items():
    print(f"{name:60s} {tuple(tensor.shape)}")

print("\n=== input_proj parameters ===")
for name, tensor in checkpoint['input_proj'].items():
    print(f"{name:60s} {tuple(tensor.shape)}")
