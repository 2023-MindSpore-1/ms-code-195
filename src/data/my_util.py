import math
from PIL import Image, ImageOps, ImageEnhance
from mindspore.dataset.vision import Inter
import numpy
import random
import cv2

max_value = 10.

def resample():
    return random.choice((Inter.BILINEAR, Inter.BICUBIC))


def rotate(image, magnitude):
    magnitude = (magnitude / max_value) * 90

    if random.random() > 0.5:
        magnitude *= -1

    return image.rotate(magnitude, resample=resample())


def shear_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), resample=resample())


def shear_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.3

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), resample=resample())


def translate_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.5

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def identity(image, _):
    return image


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Brightness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Brightness(image).enhance(magnitude)


def color(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Color(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Color(image).enhance(magnitude)


def contrast(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Contrast(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Contrast(image).enhance(magnitude)


def sharpness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Sharpness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Sharpness(image).enhance(magnitude)


def solar(image, magnitude):
    magnitude = int((magnitude / max_value) * 256)
    if random.random() > 0.5:
        return ImageOps.solarize(image, magnitude)
    else:
        return ImageOps.solarize(image, 256 - magnitude)


def poster(image, magnitude):
    magnitude = int((magnitude / max_value) * 4)
    if random.random() > 0.5:
        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)
    else:
        if random.random() > 0.5:
            magnitude = 4 - magnitude
        else:
            magnitude = 4 + magnitude

        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)


def random_hsv(image):
    x = numpy.arange(0, 256, dtype=numpy.int16)
    hsv = numpy.random.uniform(-1, 1, 3) * [.015, .7, .4] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    lut_hue = ((x * hsv[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * hsv[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * hsv[2], 0, 255).astype('uint8')

    h = cv2.LUT(h, lut_hue)
    s = cv2.LUT(s, lut_sat)
    v = cv2.LUT(v, lut_val)

    image_hsv = cv2.merge((h, s, v)).astype('uint8')
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)


class RandomAugment:
    def __init__(self, mean=9, sigma=0.5, n=2):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, image):
        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))

            image = transform(image, magnitude)
        return image


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        i, j, h, w = self.params(image.size)
        image = image.crop((j, i, j + w, i + h))
        return image.resize([self.size, self.size], resample())

    @staticmethod
    def params(size):
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        for _ in range(10):
            target_area = random.uniform(*scale) * size[0] * size[1]
            aspect_ratio = math.exp(random.uniform(*(math.log(ratio[0]), math.log(ratio[1]))))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= size[0] and h <= size[1]:
                i = random.randint(0, size[1] - h)
                j = random.randint(0, size[0] - w)
                return i, j, h, w

        in_ratio = size[0] / size[1]
        if in_ratio < min(ratio):
            w = size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = size[1]
            w = int(round(h * max(ratio)))
        else:
            w = size[0]
            h = size[1]
        i = (size[1] - h) // 2
        j = (size[0] - w) // 2
        return i, j, h, w




# """Data augment"""
# import logging
# import random
#
# import numpy as np
# import PIL
# import PIL.ImageOps
# import PIL.ImageEnhance
# import PIL.ImageDraw
# from PIL import Image
# from mindspore import Tensor
#
# logger = logging.getLogger(__name__)
#
# PARAMETER_MAX = 10
#
# def Rotate(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     print(img.shape)
#     if random.random() < 0.5:
#         v = -v
#     return img.rotate(v)
#
#
# def ShearX(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))
#
#
# def ShearY(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
#
#
# def TranslateX(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     v = int(v * img.size[0])
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
#
#
# def TranslateY(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     v = int(v * img.size[1])
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
#
#
# def AutoContrast(img, **kwarg):
#     return PIL.ImageOps.autocontrast(img)
#
#
# def Brightness(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Brightness(img).enhance(v)
#
#
# def Color(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Color(img).enhance(v)
#
#
# def Contrast(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Contrast(img).enhance(v)
#
#
# def Cutout(img, v, max_v, bias=0):
#     if v == 0:
#         return img
#     v = _float_parameter(v, max_v) + bias
#     v = int(v * min(img.size))
#     return CutoutAbs(img, v)
#
#
# def CutoutAbs(img, v, **kwarg):
#     w, h = img.size
#     x0 = np.random.uniform(0, w)
#     y0 = np.random.uniform(0, h)
#     x0 = int(max(0, x0 - v / 2.))
#     y0 = int(max(0, y0 - v / 2.))
#     x1 = int(min(w, x0 + v))
#     y1 = int(min(h, y0 + v))
#     xy = (x0, y0, x1, y1)
#     # gray
#     color = (127, 127, 127)
#     img = img.copy()
#     PIL.ImageDraw.Draw(img).rectangle(xy, color)
#     return img
#
#
# def Equalize(img, **kwarg):
#     return PIL.ImageOps.equalize(img)
#
#
# def Identity(img, **kwarg):
#     return img
#
#
# def Invert(img, **kwarg):
#     return PIL.ImageOps.invert(img)
#
#
# def Posterize(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     return PIL.ImageOps.posterize(img, v)
#
#
#
# def Sharpness(img, v, max_v, bias=0):
#     v = _float_parameter(v, max_v) + bias
#     return PIL.ImageEnhance.Sharpness(img).enhance(v)
#
#
#
# def Solarize(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     return PIL.ImageOps.solarize(img, 256 - v)
#
#
# def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
#     v = _int_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     img_np = np.array(img).astype(np.int)
#     img_np = img_np + v
#     img_np = np.clip(img_np, 0, 255)
#     img_np = img_np.astype(np.uint8)
#     img = Image.fromarray(img_np)
#     return PIL.ImageOps.solarize(img, threshold)
#
#
#
# def _float_parameter(v, max_v):
#     return float(v) * max_v / PARAMETER_MAX
#
#
# def _int_parameter(v, max_v):
#     return int(v * max_v / PARAMETER_MAX)
#
#
# def fixmatch_augment_pool():
#     # FixMatch paper
#     augs = [(AutoContrast, None, None),
#             (Brightness, 0.9, 0.05),
#             (Color, 0.9, 0.05),
#             (Contrast, 0.9, 0.05),
#             (Equalize, None, None),
#             (Identity, None, None),
#             (Posterize, 4, 4),
#             (Rotate, 30, 0),
#             (Sharpness, 0.9, 0.05),
#             (ShearX, 0.3, 0),
#             (ShearY, 0.3, 0),
#             (Solarize, 256, 0),
#             (TranslateX, 0.3, 0),
#             (TranslateY, 0.3, 0)]
#     return augs
#
#
# def my_augment_pool():
#     # Test
#     augs = [(AutoContrast, None, None),
#             (Brightness, 1.8, 0.1),
#             (Color, 1.8, 0.1),
#             (Contrast, 1.8, 0.1),
#             (Cutout, 0.2, 0),
#             (Equalize, None, None),
#             (Invert, None, None),
#             (Posterize, 4, 4),
#             (Rotate, 30, 0),
#             (Sharpness, 1.8, 0.1),
#             (ShearX, 0.3, 0),
#             (ShearY, 0.3, 0),
#             (Solarize, 256, 0),
#             (SolarizeAdd, 110, 0),
#             (TranslateX, 0.45, 0),
#             (TranslateY, 0.45, 0)]
#     return augs
#
#
# class RandAugmentPC():
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = my_augment_pool()
#
#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             prob = np.random.uniform(0.2, 0.8)
#             if random.random() + prob >= 1:
#                 img = op(img, v=self.m, max_v=max_v, bias=bias)
#         img = CutoutAbs(img, int(32*0.5))
#         return img
#
#
# class RandAugmentMC():
#     def __init__(self, n, m):
#         assert n >= 1
#         assert 1 <= m <= 10
#         self.n = n
#         self.m = m
#         self.augment_pool = fixmatch_augment_pool()
#
#     def __call__(self, img):
#         ops = random.choices(self.augment_pool, k=self.n)
#         for op, max_v, bias in ops:
#             v = np.random.randint(1, self.m)
#             if random.random() < 0.5:
#                 img = op(img, v=v, max_v=max_v, bias=bias)
#                 # print(op)
#         img = CutoutAbs(img, int(32*0.5))
#         return img