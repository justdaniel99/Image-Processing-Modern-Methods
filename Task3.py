import argparse
import cv2
import numpy as np
from scipy import ndimage


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # grad (sigma)
    grad_parser = subparsers.add_parser("grad")
    grad_parser.add_argument("params", nargs=1)

    # nonmax(sigma)
    nonmax_parser = subparsers.add_parser("nonmax")
    nonmax_parser.add_argument("params", nargs=1)

    # canny (sigma) (thr_high) (thr_low)
    canny_parser = subparsers.add_parser("canny")
    canny_parser.add_argument("params", nargs=3)

    # vessels
    subparsers.add_parser("vessels")

    parser.add_argument("input_file", nargs=1)
    parser.add_argument("output_file", nargs=1)

    return parser


# image file reading
def image_file_read(image_file):
    return cv2.imread(image_file)


# image file saving
def image_file_save(image_file_path, image_file):
    cv2.imwrite(image_file_path, image_file)


# gradient (sigma)
def grad(params, input_image, output_image):
    sigma = float(params[0])

    img = image_file_read(input_image).astype(float)
    Ix = np.zeros((img.shape[0], img.shape[1], 3))
    Iy = np.zeros((img.shape[0], img.shape[1], 3))
    G = np.zeros((img.shape[0], img.shape[1], 3))

    size = 2 * round(3 * sigma) + 1
    x = np.arange(-(size - 1) // 2, (size + 1) // 2)
    y = np.arange(-(size - 1) // 2, (size + 1) // 2).reshape(-1, 1)
    gx = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- x / sigma ** 2)
    gy = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- y / sigma ** 2)

    for i in range(3):
        Ix[:, :, i] = ndimage.convolve(img[:, :, i], gx)
        Iy[:, :, i] = ndimage.convolve(img[:, :, i], gy)
        G[:, :, i] = np.hypot(Ix[:, :, i], Iy[:, :, i])

    G = G / G.max() * 255
    image_file_save(output_image, G.astype('uint8'))


# nonmax (without file saving)
def nonmax_without_saving(input_image, sigma):
    img = image_file_read(input_image).astype(float)
    Ix = np.zeros((img.shape[0], img.shape[1], 3))
    Iy = np.zeros((img.shape[0], img.shape[1], 3))
    G = np.zeros((img.shape[0], img.shape[1], 3))
    Z = np.zeros((img.shape[0], img.shape[1], 3))

    size = 2 * round(3 * sigma) + 1
    x = np.arange(- (size - 1) // 2, (size + 1) // 2)
    y = np.arange(- (size - 1) // 2, (size + 1) // 2).reshape(-1, 1)
    y = np.flip(y)
    gx = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- x / sigma ** 2)
    gy = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- y / sigma ** 2)

    for i in range(3):
        Ix[:, :, i] = ndimage.convolve(img[:, :, i], gx)
        Iy[:, :, i] = ndimage.convolve(img[:, :, i], gy)
        G[:, :, i] = np.hypot(Ix[:, :, i], Iy[:, :, i])

    G = G / G.max() * 255

    theta = np.arctan2(Iy, Ix)

    theta = theta * 180. / np.pi
    theta[theta < 0] += 180

    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):
            for k in range(3):

                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0.0 <= theta[i, j, k] < 22.5) or (157.5 <= theta[i, j, k] <= 180.0):
                        q = G[i, j + 1, k]
                        r = G[i, j - 1, k]

                    # angle 45
                    elif 22.5 <= theta[i, j, k] < 67.5:
                        q = G[i + 1, j - 1, k]
                        r = G[i - 1, j + 1, k]

                    # angle 90
                    elif 67.5 <= theta[i, j, k] < 112.5:
                        q = G[i + 1, j, k]
                        r = G[i - 1, j, k]

                    # angle 135
                    elif 112.5 <= theta[i, j, k] < 157.5:
                        q = G[i - 1, j - 1, k]
                        r = G[i + 1, j + 1, k]

                    if (G[i, j, k] >= q) and (G[i, j, k] >= r):
                        Z[i, j, k] = G[i, j, k]
                    else:
                        Z[i, j, k] = 0

                except IndexError:
                    pass

    return Z


# nonmax (sigma)
def nonmax(params, input_image, output_image):
    sigma = float(params[0])

    image_file_save(output_image, nonmax_without_saving(input_image, sigma).astype("uint8"))


# canny (sigma) (thr_high) (thr_low)
def canny(params, input_image, output_image):
    sigma = float(params[0])
    highThresholdRatio = float(params[1])
    lowThresholdRatio = float(params[2])

    img = nonmax_without_saving(input_image, sigma)

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    res = np.zeros((img.shape[0], img.shape[1], 3))

    weak = np.int32(25)
    strong = np.int32(255)

    for i in range(3):
        strong_i, strong_j = np.where(img[:, :, i] >= highThreshold)

    for i in range(3):
        weak_i, weak_j = np.where((img[:, :, i] <= highThreshold) & (img[:, :, i] >= lowThreshold))

    for i in range(3):
        res[strong_i, strong_j, i] = strong
        res[weak_i, weak_j, i] = weak

    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            for k in range(3):
                if res[i, j, k] == weak:
                    try:
                        if ((res[i + 1, j - 1, k] == strong) or (res[i + 1, j, k] == strong)
                                or (res[i + 1, j + 1, k] == strong) or (res[i, j - 1, k] == strong)
                                or (res[i, j + 1, k] == strong) or (res[i - 1, j - 1, k] == strong)
                                or (res[i - 1, j, k] == strong) or (res[i - 1, j + 1, k] == strong)):
                            res[i, j, k] = strong
                        else:
                            res[i, j, k] = 0
                    except IndexError:
                        pass

    image_file_save(output_image, res.astype('uint8'))


# vessels [for chosen sigma]
def vessels_sigma(img, sigma):
    convolvexx = np.zeros((img.shape[0], img.shape[1], 3))
    convolvexy = np.zeros((img.shape[0], img.shape[1], 3))
    convolveyy = np.zeros((img.shape[0], img.shape[1], 3))
    G = np.zeros((img.shape[0], img.shape[1], 3))
    Z = np.zeros((img.shape[0], img.shape[1], 3))
    theta = np.zeros((img.shape[0], img.shape[1], 3))

    size = 2 * round(3 * sigma) + 1

    x = np.arange(- (size - 1) // 2, (size + 1) // 2)
    y = np.arange(- (size - 1) // 2, (size + 1) // 2).reshape(-1, 1)

    gxx = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- 1 / sigma ** 2 + x ** 2 / sigma ** 4)
    gxy = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (x * y / sigma ** 4)
    gyy = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2)) * (- 1 / sigma ** 2 + y ** 2 / sigma ** 4)

    convolvexx[:, :, 0] = ndimage.convolve(img[:, :, 0], gxx)
    convolvexy[:, :, 0] = ndimage.convolve(img[:, :, 0], gxy)
    convolveyy[:, :, 0] = ndimage.convolve(img[:, :, 0], gyy)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):

            hessian = np.block([[np.sum(convolvexx[i - 1:i + 1, j - 1:j + 1, 0]),
                                 np.sum(convolvexy[i - 1:i + 1, j - 1:j + 1, 0])],
                                [np.sum(convolvexy[i - 1:i + 1, j - 1:j + 1, 0]),
                                 np.sum(convolveyy[i - 1:i + 1, j - 1:j + 1, 0])]])

            eigenvalues, eigenvectors = np.linalg.eig(hessian)
            abs_eigenvalues = np.abs(eigenvalues)
            index = np.argmax(abs_eigenvalues)

            theta[i, j, 0] = np.arctan2(eigenvectors[index][1], eigenvectors[index][0]) * 180 / np.pi

            if eigenvalues[index] > 0:
                G[i, j, 0] = eigenvalues[index]
            else:
                G[i, j, 0] = 0

    G = G / G.max() * 255
    theta[theta < 0] += 180

    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):

            try:
                q = 255
                r = 255

                # angle 0
                if (0.0 <= theta[i, j, 0] < 22.5) or (157.5 <= theta[i, j, 0] <= 180.0):
                    q = G[i, j + 1, 0]
                    r = G[i, j - 1, 0]

                # angle 45
                elif 22.5 <= theta[i, j, 0] < 67.5:
                    q = G[i + 1, j - 1, 0]
                    r = G[i - 1, j + 1, 0]

                # angle 90
                elif 67.5 <= theta[i, j, 0] < 112.5:
                    q = G[i + 1, j, 0]
                    r = G[i - 1, j, 0]

                # angle 135
                elif 112.5 <= theta[i, j, 0] < 157.5:
                    q = G[i - 1, j - 1, 0]
                    r = G[i + 1, j + 1, 0]

                if (G[i, j, 0] >= q) and (G[i, j, 0] >= r):
                    Z[i, j, 0] = G[i, j, 0]
                else:
                    Z[i, j, 0] = 0

            except IndexError:
                pass

    for i in range(3):
        Z[:, :, i] = Z[:, :, 0]

    return Z


# vessels [combined function]
def vessels(input_image, output_image):
    img = image_file_read(input_image).astype(float)

    final = np.zeros((img.shape[0], img.shape[1], 3))

    img_sigma_2 = vessels_sigma(img, 2)
    img_sigma_3 = vessels_sigma(img, 3)
    img_sigma_4 = vessels_sigma(img, 4)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            final[i, j, 0] = max(img_sigma_2[i, j, 0], img_sigma_3[i, j, 0], img_sigma_4[i, j, 0])

    for k in range(3):
        final[:, :, k] = final[:, :, 0]

    image_file_save(output_image, final.astype('uint8'))


if __name__ == '__main__':
    namespace = create_parser().parse_args()

    if namespace.command == "grad":
        grad(namespace.params, namespace.input_file[0], namespace.output_file[0])
    elif namespace.command == "nonmax":
        nonmax(namespace.params, namespace.input_file[0], namespace.output_file[0])
    elif namespace.command == "canny":
        canny(namespace.params, namespace.input_file[0], namespace.output_file[0])
    else:
        vessels(namespace.input_file[0], namespace.output_file[0])

