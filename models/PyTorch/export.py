import torch
from torchvision import models

model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1
).eval()

example_input = torch.randn(1, 3, 224, 224)

traced_model = torch.jit.trace(
    model,
    example_input
)

traced_model.save("1/model.pt")

print("PyTorch model exported successfully to 1/model.pt")