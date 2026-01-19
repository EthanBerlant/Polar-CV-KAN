import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        print(f"Tensor on GPU: {x}")
        print("GPU tensor operation successful!")
    except Exception as e:
        print(f"GPU tensor operation failed: {e}")
else:
    print("CUDA not available")
