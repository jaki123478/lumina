import torch

from lumina_image_experimental import LuminaImageDenoiser, create_model


def test_lumina_image_forward_shape():
    model = create_model("cpu")
    noisy = torch.randn(2, 3, 16, 16)
    tokens = torch.zeros(2, 4, dtype=torch.long)
    steps = torch.zeros(2, dtype=torch.long)
    output = model(noisy, tokens, steps)
    assert output.shape == noisy.shape


def test_lumina_image_parameters_are_trainable():
    model = LuminaImageDenoiser()
    assert sum(parameter.numel() for parameter in model.parameters()) > 1_000_000
    assert all(parameter.requires_grad for parameter in model.parameters())
