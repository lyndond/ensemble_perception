import torch
import numpy as np
import src.tools.image as ta


class TestToolsImage():
    def test_tmp(self):
        unprocessed_img = ta.unprocess(torch.rand((1, 3, 256, 256)))
        assert unprocessed_img.shape == (256, 256, 3)
        assert isinstance(unprocessed_img, np.ndarray)
        assert np.all(unprocessed_img <= 1) and np.all(unprocessed_img>=0)
        print(np.max(unprocessed_img), np.min(unprocessed_img))

    def test_vec_to_image(self):
        assert ta.vec_to_image(torch.rand(256*256*3)).shape == (1, 3, 256, 256)

        # specify chan, height, width
        assert ta.vec_to_image(torch.rand(10*12*2), c=2, h=10, w=12).shape == (1, 2, 10, 12)

    def test_clamp(self):
        clamped = ta.clamp(torch.randn(10000), min_val=0., max_val=1.)
        assert torch.all(clamped <= 1.) and torch.all(clamped >= 0)

        clamped_np = ta.clamp(np.random.randn(10000), min_val=0., max_val=1.)
        assert np.all(clamped_np <= 1.) and np.all(clamped_np >= 0.)

    def test_rescale(self):
        rescaled = ta.rescale(torch.randn(10000))
        assert torch.all(rescaled <= 1.) and torch.all(rescaled >= 0)

        rescaled_np = ta.rescale(np.random.randn(10000))
        assert np.all(rescaled_np <= 1.) and np.all(rescaled_np >= 0.)
