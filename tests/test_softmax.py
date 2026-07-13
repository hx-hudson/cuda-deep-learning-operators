import torch
import custom_ops

def test():

    rows = 4096
    cols = [128, 256, 512, 1024, 2048, 4096]

    for col in cols:
        x = torch.randn(
            rows, col, device="cuda", dtype=torch.float32)
        
        y = custom_ops.softmax(x)
        expect = torch.softmax(x, dim=-1)

        torch.cuda.synchronize()

        torch.testing.assert_close(
            y,
            expect,
            rtol=1e-4,
            atol=1e-5,
        )

        row_sums = y.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-4,
        )

        print("Input shape:", tuple(x.shape))
        print("Input device:", x.device)
        print("Input dtype:", x.dtype)
        print("Maximum difference:",
            (y - expect).abs().max().item())
        print("softmax correctness check passed!")

if __name__ == "__main__":
    test()
