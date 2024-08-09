"""
This module contains transform classes to apply affine transformations to images.
The transformation can be random or fixed.
Including specific transformations for rotations.

"""
import warnings
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _get_inverse_affine_matrix

from invertransforms import functional as F
from invertransforms.lib import InvertibleError, Invertible


class Affine(Invertible):
    """
    Apply affine transformation on the image.

    Args:
        Almost the same as torchvision.transforms.RandomAffine,
        with the only difference that arguments {angle, translate, scale, shear}
        are deterministic, typically the return of `RandomAffine.get_params`.
    """

    def __init__(self, angle, translate, scale, shear, interpolation=InterpolationMode.NEAREST, fill=None, center=None):
        self.params = (angle, translate, scale, shear)
        self.interpolation = interpolation
        self.fill = fill
        self.center = center

    def inverse(self):
        return Affine(*_invert_affine_params(*self.params), self.interpolation, self.fill, self.center)

    def __call__(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        return F.affine(img, *self.params, interpolation=self.interpolation, fill=fill, center=self.center)

    def __repr__(self):
        return f'{self.__class__.__name__}(params={self.params}, interpolation={self.interpolation})'


class RandomAffine(transforms.RandomAffine, Invertible):
    def __init__(self, *args, **kwargs):
        transforms.RandomAffine.__init__(self, *args, **kwargs)
        self._params = None

        # For invertability, specify only 1 affine transform at a time, e.g.
        # ```
        # Compose([
        #     RandomAffine(degrees=45),
        #     RandomAffine(degrees=0, translate=(0.1, 0.1)),
        #     RandomAffine(degrees=0, translate=None, scale=(0.5, 2)),
        #     RandomAffine(degrees=0, translate=None, scale=None, shear=(-5, 10, -10, 5)),
        # ])
        # ```
        n_trfm = sum([
            any([abs(x) > 1e-7 for x in self.degrees]),
            self.translate is not None,
            self.scale is not None,
            self.shear is not None
        ])
        if n_trfm > 1:
            warnings.warn("Too much affine transforms specified at a time! "
                "For better invertability, it is suggested that specifying only one of "
                "{degrees, translate, scale, shear} at a time, and then compose them with `Compose`."
            )

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        img_size = [width, height]  # flip for keeping BC on get_params call

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        self._params = ret # record params for (potential) inversion

        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        return Affine(*_invert_affine_params(*self._params), self.interpolation, self.fill, self.center)

    def _can_invert(self):
        return self._params is not None


def _invert_affine_params(angle, translations, scale, shear):
    new_angle = - angle
    assert -180 <= new_angle <= 180
    new_translations = tuple(-t for t in translations)
    new_scale = 1.0 / scale
    new_shear = tuple(-s for s in shear)
    return new_angle, new_translations, new_scale, new_shear


class Rotation(Invertible):
    """
    Rotate the image given an angle (in degrees).

    Args:
        angle (float or int): degrees of the angle
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, angle, resample=False, expand=False, center=None):
        self.angle = angle
        self.resample = resample
        self.expand = expand
        self.center = center
        self._img_h = self._img_w = None

    def __call__(self, img):
        first_call = self._img_h is None or self._img_w is None
        if first_call:
            self._img_w, self._img_h = img.size
        img = F.rotate(img, self.angle, self.resample, self.expand, self.center)
        if not first_call and self.expand:
            img = F.center_crop(img=img, output_size=(self._img_h, self._img_w))
        return img

    def inverse(self):
        if (self._img_h is None or self._img_w is None) and self.expand:
            raise InvertibleError(
                'Cannot invert a transformation before it is applied'
                ' (size of image before expanded rotation unknown).')  # note: the size could be computed
        rot = Rotation(
            angle=-self.angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self._img_h, self._img_w
        return rot

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angle={0}'.format(self.angle)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomRotation(transforms.RandomRotation, Invertible):
    _angle = _img_h = _img_w = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        self._angle = self.get_params(self.degrees)
        self._img_w, self._img_h = img.size
        return F.rotate(img, self._angle, self.resample, self.expand, self.center)

    def inverse(self):
        if not self._can_invert():
            raise InvertibleError('Cannot invert a random transformation before it is applied.')

        rot = Rotation(
            angle=-self._angle,
            resample=self.resample,
            expand=self.expand,
            center=self.center,
        )
        rot._img_h, rot._img_w = self._img_h, self._img_w
        return rot

    def _can_invert(self):
        return self._angle is not None and self._img_w is not None and self._img_h is not None
