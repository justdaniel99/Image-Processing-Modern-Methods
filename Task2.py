import argparse
from math import log10, sqrt
import cv2
import numpy as np
from scipy import ndimage


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # mse
    subparsers.add_parser("mse")

    # psnr
    subparsers.add_parser("psnr")

    # ssim
    subparsers.add_parser("ssim")

    # median (rad)
    median_parser = subparsers.add_parser("median")
    median_parser.add_argument("params", nargs=1)

    # gauss (sigma_d)
    gauss_parser = subparsers.add_parser("gauss")
    gauss_parser.add_argument("params", nargs=1)

    # bilateral (sigma_d) (sigma_r)
    bilateral_parser = subparsers.add_parser("bilateral")
    bilateral_parser.add_argument("params", nargs=2)

    parser.add_argument("input_file", nargs=1)
    parser.add_argument("output_file", nargs=1)

    return parser


# image file reading
def image_file_read(image_file):
    return cv2.imread(image_file)


# image file saving
def image_file_save(image_file_path, image_file):
    cv2.imwrite(image_file_path, image_file)


# covariance of two numbers
def covariance(x, y):
    x_bar, y_bar = x.mean(), y.mean()
    return np.sum((x - x_bar) * (y - y_bar)) / np.size(x)


# mse
def mse(image1_path, image2_path):
    image1 = image_file_read(image1_path).astype("float")
    image2 = image_file_read(image2_path).astype("float")
    err = np.sum((image1 - image2) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


# psnr
def psnr(image1_path, image2_path):
    mse_number = mse(image1_path, image2_path)
    if mse_number == 0:
        return 100
    return 20 * log10(255.0 / sqrt(mse_number))


# ssim
def ssim(image1_path, image2_path):
    l_const = 255.0
    image1 = image_file_read(image1_path).astype("float")
    image2 = image_file_read(image2_path).astype("float")
    mu_x = np.mean(image1)
    mu_y = np.mean(image2)
    c1 = (0.01 * l_const) ** 2
    c2 = (0.03 * l_const) ** 2
    return (2 * mu_y * mu_x + c1) * (2 * covariance(image1, image2) + c2) / ((mu_x ** 2 + mu_y ** 2 + c1) *
                                                                             (np.var(image2) + np.var(image1) + c2))


# median (rad)
def median(params, input_image_path, output_image_path):
    r = int(params[0])

    img = np.array(image_file_read(input_image_path))
    new_img = np.zeros((img.shape[0], img.shape[1], 3))
    h, w = img.shape[:2]

    for i in range(h):
        for j in range(w):
            new_img[i, j, 0] = np.median(img[max(0, i - r): min(h, i + r + 1), max(0, j - r): min(w, j + r + 1), 0])
            new_img[i, j, 1] = np.median(img[max(0, i - r): min(h, i + r + 1), max(0, j - r): min(w, j + r + 1), 1])
            new_img[i, j, 2] = np.median(img[max(0, i - r): min(h, i + r + 1), max(0, j - r): min(w, j + r + 1), 2])

    image_file_save(output_image_path, new_img.astype('uint8'))


# gauss (sigma_d)
def gauss(params, input_image_file, output_image_file):
    sigma_d = float(params[0])

    img = image_file_read(input_image_file)
    new_img = np.zeros((img.shape[0], img.shape[1], 3))

    # gauss matrix
    size = 2 * round(3 * sigma_d) + 1
    x = np.arange(-(size - 1) // 2, (size + 1) // 2)
    y = np.arange(-(size - 1) // 2, (size + 1) // 2).reshape(-1, 1)
    f = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_d ** 2))
    m = f / np.sum(f)

    new_img[:, :, 0] = ndimage.convolve(img[:, :, 0], m)
    new_img[:, :, 1] = ndimage.convolve(img[:, :, 1], m)
    new_img[:, :, 2] = ndimage.convolve(img[:, :, 2], m)

    # edges processing
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0

    image_file_save(output_image_file, new_img.astype('uint8'))


# extending image with given size on edges
def extend(size, img):
    new_img = np.zeros((img.shape[0] + 2 * size, img.shape[1] + 2 * size, 3))
    new_img[size:img.shape[0] + size, size:img.shape[1] + size, ] = img.copy()

    for i in range(size):
        for j in range(img.shape[0]):
            new_img[j + size][i] = new_img[j + size][size]
            new_img[j + size][i + size + img.shape[1]] = new_img[j + size][size + img.shape[1] - 1]

    for i in range(size):
        for j in range(new_img.shape[1]):
            new_img[i][j] = new_img[size][j]
            new_img[size + img.shape[0] + i][j] = new_img[size + img.shape[0] - 1][j]

    return new_img


# bilateral (sigma_d) (sigma_r)
def bilateral(params, input_file_name, output_file_name):
    sigma_d = float(params[0])
    sigma_r = float(params[1])

    size = round(3 * sigma_d)
    window = 2 * size + 1

    img = image_file_read(input_file_name)
    extended = extend(size, img).astype(float)

    gaussian_d = np.zeros((size * 2 + 1, size * 2 + 1, 3))
    for i in range(size * 2 + 1):
        for j in range(size * 2 + 1):
            gaussian_d[i, j, :] = np.exp(- (((i - size) ** 2) + (j - size) ** 2) / (2 * sigma_d ** 2)) / \
                                  (2 * sigma_d ** 2 * np.pi)

    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            image_window = extended[i:i + window, j:j + window]
            gaussian_r = np.exp(- (image_window - image_window[size, size]) ** 2 / (2 * sigma_r ** 2))
            gaussian_kernel = gaussian_r * gaussian_d
            res[i, j, :] = np.full(3, np.sum(image_window * gaussian_kernel / gaussian_kernel.sum()))

    image_file_save(output_file_name, res.astype('uint8'))


if __name__ == '__main__':
    namespace = create_parser().parse_args()

    if namespace.command == "mse":
        print(mse(namespace.input_file[0], namespace.output_file[0]))
    elif namespace.command == "psnr":
        print(psnr(namespace.input_file[0], namespace.output_file[0]))
    elif namespace.command == "ssim":
        print(ssim(namespace.input_file[0], namespace.output_file[0]))
    elif namespace.command == "median":
        median(namespace.params, namespace.input_file[0], namespace.output_file[0])
    elif namespace.command == "gauss":
        gauss(namespace.params, namespace.input_file[0], namespace.output_file[0])
    else:
        bilateral(namespace.params, namespace.input_file[0], namespace.output_file[0])
