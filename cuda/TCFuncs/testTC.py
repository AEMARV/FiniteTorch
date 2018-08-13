import tensor_comprehensions as tc
import torch
lang = """
def tensordot(float(B, C, H, W) I, float(N, C, H1, W1) F) -> (O(B,N,H,W)) {
    O(b,n, h, w) +=! I(n, c1, h+h1, w+w1) + F(n, c1, h1, w1)
}
"""
N, C1, C2, C3, H, W = 32, 512, 8, 2, 28, 28
tensordot = tc.define(lang, name="tensordot")
I0, I1 = torch.randn(N, C1, C2, H, W).cuda(), torch.randn(N, C2, C3, H, W).cuda()
best_options = tensordot.autotune(I0, I1, cache=True)
out = tensordot(I0, I1, options=best_options)