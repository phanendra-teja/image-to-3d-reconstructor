"""Phase 0 environment check: Python, PyTorch, CUDA, and Depth Anything V2 Small."""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA not available: PyTorch is probably a CPU-only build. Fix this first.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("Total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

# Downloads ~100 MB on first run into the Hugging Face cache (outside this repo).
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

# Random noise image: ONLY a compatibility test. Its depth values are meaningless.
dummy = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
inputs = processor(images=dummy, return_tensors="pt").to(device)
print("Model input shape:", tuple(inputs["pixel_values"].shape))

with torch.no_grad():
    out = model(**inputs)

print("Predicted depth shape:", tuple(out.predicted_depth.shape))
print("Peak VRAM allocated (MB):", round(torch.cuda.max_memory_allocated() / 1e6, 1))
print("Environment check passed.")