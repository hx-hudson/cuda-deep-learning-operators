import torch
import custom_ops

def test_relu():
    torch.manual_seed(42)

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    expect = torch.relu(x)
    y = custom_ops.relu(x)

    torch.cuda.synchronize()

    torch.testing.assert_close(y, expect)

    print("Input shape:", tuple(x.shape))
    print("Input device:", x.device)
    print("Input dtype:", x.dtype)
    print("Maximum difference:",
          (y - expect).abs().max().item())
    print("ReLU correctness check passed!")

if __name__ == "__main__":
    test_relu()