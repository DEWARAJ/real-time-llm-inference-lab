import torch

from inference_lab.kernels import rms_norm, rms_norm_reference


def test_rmsnorm_fallback_matches_reference():
    torch.manual_seed(3)
    x = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    expected = rms_norm_reference(x, weight)
    actual = rms_norm(x, weight)
    torch.testing.assert_close(actual, expected)
