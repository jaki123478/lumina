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


def test_lumina_image_rectified_flow_and_offline_gen(tmp_path):
    from lumina_image_experimental import generate_rectified_flow
    from lumina_image_gen import generate_image

    model = create_model("cpu", channels=32)
    rgb = generate_rectified_flow(model, "tramonto sul mare", steps=4, size=32, cfg_scale=2.0, seed=7)
    assert rgb.shape == (3, 32, 32)
    assert torch.isfinite(rgb).all()
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    ckpt = tmp_path / "test_img_ckpt.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_size": 32000, "channels": 32, "size": 32}, ckpt)
    out_png = tmp_path / "out.png"
    import lumina_image_gen

    old_model = lumina_image_gen._NATIVE_IMAGE_MODEL
    old_size = lumina_image_gen._NATIVE_IMAGE_SIZE
    try:
        lumina_image_gen._NATIVE_IMAGE_MODEL = model.eval()
        lumina_image_gen._NATIVE_IMAGE_SIZE = 32
        saved = generate_image("un gatto arancione", output_path=str(out_png), steps=3, out_size=64)
        assert out_png.exists()
        assert saved == str(out_png)
        assert out_png.stat().st_size > 100
    finally:
        lumina_image_gen._NATIVE_IMAGE_MODEL = old_model
        lumina_image_gen._NATIVE_IMAGE_SIZE = old_size
