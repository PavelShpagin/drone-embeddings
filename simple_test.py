print("Hello World!")
print("Testing Python functionality")

try:
    import torch
    print("PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"Error importing PyTorch: {e}")

try:
    import timm
    print("timm imported successfully")
    print(f"timm version: {timm.__version__}")
except Exception as e:
    print(f"Error importing timm: {e}")

print("Test complete!") 