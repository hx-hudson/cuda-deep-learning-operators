import torch
import custom_ops

def test():

    rows = 4096
    cols = [128, 256, 512, 1024, 2048, 4096]

    for col in cols:

        print(f"Testing rows={rows}, cols={col}")

        x = torch.randn(
            rows, col, device="cuda", dtype=torch.float32)
        
        gamma = torch.randn(col, device="cuda", dtype=torch.float32)
        beta = torch.randn(col, device="cuda", dtype=torch.float32)
        
        y = custom_ops.layernorm(x, gamma, beta, eps=1e-5)
        expect = torch.layer_norm(x, (col,), gamma, beta, eps=1e-5)

        torch.cuda.synchronize()

        torch.testing.assert_close(
            y,
            expect,
            rtol=1e-4,
            atol=1e-5,
        )

        print("Input shape:", tuple(x.shape))
        print("Maximum difference:",
            (y - expect).abs().max().item())
        print("layernorm correctness check passed!")

if __name__ == "__main__":
    test()