import torch
import custom_ops

def test():

    NMK = [(512, 256, 1024), (1024, 2048, 512), (257, 129, 513)]

    for N, M, K in NMK:

        print(f"N = {N}, M = {M}, K = {K}")

        a = torch.randn(N, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, M, device="cuda", dtype=torch.float32)

        c = custom_ops.matmul(a, b)
        expect = torch.matmul(a, b)

        torch.cuda.synchronize()

        torch.testing.assert_close(
            c,
            expect,
            rtol=1e-3,
            atol=1e-3,
        )

        print("matmul correctness check passed!")

if __name__ == "__main__":
    test()