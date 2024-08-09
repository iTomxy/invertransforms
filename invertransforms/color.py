"""
This modules contains transformations on the image channels (RGB, grayscale).

Technically these transformations cannot be inverted or it simply makes not much sense,
hence the inverse is usually the identity function.
"""
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import invertransforms as T
from invertransforms.lib import InvertibleError, Invertible, flip_coin, PseudoInversion, TogglePseudoInversion


class DeterministicColorJitter(Invertible):
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None, fn_idx=torch.randperm(4)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.fn_idx = fn_idx

    def inverse(self):
        return DeterministicColorJitter(
            brightness=None if self.brightness is None else 1.0 / self.brightness,
            contrast=None if self.contrast is None else 1.0 / self.contrast,
            saturation=None if self.saturation is None else 1.0 / self.saturation,
            hue=None if self.hue is None else - self.hue,
            fn_idx=reversed(self.fn_idx),
        )

    def __call__(self, img):
        for fn_id in self.fn_idx:
            if fn_id == 0 and self.brightness is not None:
                img = F.adjust_brightness(img, self.brightness)
            elif fn_id == 1 and self.contrast is not None:
                img = F.adjust_contrast(img, self.contrast)
            elif fn_id == 2 and self.saturation is not None:
                img = F.adjust_saturation(img, self.saturation)
            elif fn_id == 3 and self.hue is not None:
                img = F.adjust_hue(img, self.hue)

        return img

    def __repr__(self):
        s = self.__class__.__name__ + '('
        for n, p in zip(
            (self.brightness, self.contrast, self.saturation, self.hue),
            ("brightness", "contrast", "saturation", "hue"),
        ):
            if p is not None:
                s += f"{n}={p}, "
        return s.rstrip(", ") + ')'


class ColorJitter(transforms.ColorJitter, Invertible):
    """This transform can NOT be fully inverted."""

    def __init__(self, *args, **kwargs):
        transforms.ColorJitter.__init__(self, *args, **kwargs)
        self._params = self._fn_idx = self._x = self._x_aug = None

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        self._x = img
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        self._params = (brightness_factor, contrast_factor, saturation_factor, hue_factor)
        self._fn_idx = fn_idx

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        self._x_aug = img
        return img

    def inverse(self, stupid=False):
        """stupid: bool, if True, use `PseudoInversion`"""
        if stupid:
            if self._x is None or self._x_aug is None:
                raise InvertibleError('Cannot invert a random transformation before it is applied.')
            return PseudoInversion(self._x, self._x_aug)

        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return DeterministicColorJitter(
            brightness=None if self._params[0] is None else 1.0 / self._params[0],
            contrast=None if self._params[1] is None else 1.0 / self._params[1],
            saturation=None if self._params[2] is None else 1.0 / self._params[2],
            hue=None if self._params[3] is None else - self._params[3],
            fn_idx=reversed(self._fn_idx),
        )

    def _can_invert(self):
        if self._params is None or self._fn_idx is None:
            return False
        for p in self._params[:3]:
            if 0 == p:
                return False
        return True


class RandomGrayscale(transforms.RandomGrayscale, Invertible):
    def __init__(self, *args, **kwargs):
        transforms.RandomGrayscale.__init__(self, *args, **kwargs)
        self._x = self._x_aug = None

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        self._x = img
        num_output_channels, _, _ = F.get_dimensions(img)
        self._x_aug = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        if torch.rand(1) < self.p:
            return self._x_aug
        return img

    def inverse(self, stupid=False):
        """stupid: bool, if True, use `TogglePseudoInversion`"""
        if not stupid:
            raise InvertibleError('Non-invertible. Unless set `stupid=True` to enable fake inversion.')
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return TogglePseudoInversion(self._x, self._x_aug, self.p)

    def _can_invert(self):
        return self._x is not None and self._x_aug is not None
