import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("after fc1", x.dtype)
        x = self.relu(x)
        print("after relu", x.dtype)
        x = self.ln(x)
        print("after ln", x.dtype)
        x = self.fc2(x)
        print("after fc2", x.dtype)
        return x

# Set the model and inputs
model = ToyModel(512, 1).to("cuda")
dtype = torch.float16

with torch.autocast(device_type="cuda", dtype=dtype):
    # Input tensor 'x' in float16
    x = torch.rand(1, 512).to("cuda", dtype=torch.float16)  # Ensure x is in the same dtype as the model
    
    # Target tensor 'y_' with class indices (torch.long)
    y_ = torch.randint(0, 2, (1, 1)).to("cuda").to(torch.float16)  # Class indices should be in torch.long (0 or 1)
    
    # Print parameter dtypes
    for k, v in model.named_parameters():
        print(k, v.dtype)

    # Forward pass (logits)
    y = model(x)

    # Cross-entropy loss calculation (no need to cast y_)
    loss_function = torch.nn.BCEWithLogitsLoss()
    loss = loss_function(y, y_)
    print(loss.dtype)
    # Print gradient dtypes after backward pass
    loss.backward()  # Compute gradients
    for k, v in model.named_parameters():
        print(k, v.grad.dtype)  # Gradient dtype

