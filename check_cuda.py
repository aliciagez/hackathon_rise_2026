import torch

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("backends:", torch._dynamo.list_backends())

devise = device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")