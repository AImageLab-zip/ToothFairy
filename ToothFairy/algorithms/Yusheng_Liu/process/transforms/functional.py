
import copy
import random
import numpy as np
from scipy.ndimage import gaussian_filter


def random_contrast(image, contrast_range=(0.75, 1.25), preserve_range=True, p=1.0):
    if random.random() >= p:
        return image

    if np.random.random() < 0.5 and contrast_range[0] < 1:
        factor = np.random.uniform(contrast_range[0], 1)
    else:
        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

    mn = image.mean()
    if preserve_range:
        minm = image.min()
        maxm = image.max()

    image = (image - mn) * factor + mn

    if preserve_range:
        image[image < minm] = minm
        image[image > maxm] = maxm

    return image


def random_brightness_additive(image, mu=0., sigma=0.1, p=1.0):
    if random.random() >= p:
        return image
    rnd_nb = np.random.normal(mu, sigma)
    image += rnd_nb

    return image


def random_brightness_multiplicative(image, multiplier_range=(0.5, 2), p=1.0):
    if random.random() >= p:
        return image
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    image = image * multiplier

    return image


def random_gamma(image, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, p=1.0):
    if random.random() >= p:
        return image

    if invert_image:
        image = - image

    if np.random.random() < 0.5 and gamma_range[0] < 1:
        gamma = np.random.uniform(gamma_range[0], 1)
    else:
        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

    minm = image.min()
    rnge = image.max() - minm
    image = np.power(((image - minm) / float(rnge + epsilon)), gamma) * rnge + minm

    if invert_image:
        image = - image

    return image


def random_gaussian_noise(image, noise_variance=(0, 0.5), p=1.0):
    if random.random() >= p:
        return image
    variance = random.uniform(noise_variance[0], noise_variance[1])
    image = image + np.random.normal(0.0, variance, size=image.shape)

    return image


def random_gaussian_blur(image, sigma_range=(0, 0.5), p=1.0):
    if random.random() >= p:
        return image
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    image = gaussian_filter(image, sigma, order=0)

    return image


def in_painting(x, cutout_range=(0, 0.5), cnt=3, p=1.0):
    if random.random() >= p:
        return x
    image_shape = x.shape
    while cnt > 0 and random.random() < 0.95:
        block_noise_size = [random.randint(int(item * cutout_range[0]),
                                           int(item * cutout_range[1])) for item in image_shape]
        noise_start = [random.randint(3, image_shape[i] - block_noise_size[i] - 3) for i in range(3)]
        x[noise_start[0]:noise_start[0] + block_noise_size[0],
          noise_start[1]:noise_start[1] + block_noise_size[1],
          noise_start[2]:noise_start[2] + block_noise_size[2]] = np.random.rand(block_noise_size[0],
                                                                                block_noise_size[1],
                                                                                block_noise_size[2]) * 1.0
        cnt -= 1
    return x


def out_painting(x, retain_range=(0.8, 0.9), cnt=3, p=1.0):
    if random.random() >= p:
        return x
    image_shape = x.shape
    img_rows, img_cols, img_deps = image_shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(img_rows, img_cols, img_deps) * 1.0
    while cnt > 0 and random.random() < 0.95:
        block_noise_size = [random.randint(int(retain_range[0] * item),
                                           int(retain_range[1] * item)) for item in image_shape]
        noise_start = [random.randint(3, image_shape[i] - block_noise_size[i] - 3) for i in range(3)]
        retain_bbox = [noise_start[0], noise_start[0] + block_noise_size[0],
                       noise_start[1], noise_start[1] + block_noise_size[1],
                       noise_start[2], noise_start[2] + block_noise_size[2]]
        x[retain_bbox[0]:retain_bbox[1],
          retain_bbox[2]:retain_bbox[3],
          retain_bbox[4]:retain_bbox[5]] = image_temp[retain_bbox[0]:retain_bbox[1],
                                                      retain_bbox[2]:retain_bbox[3],
                                                      retain_bbox[4]:retain_bbox[5]]
        cnt -= 1

    return x


