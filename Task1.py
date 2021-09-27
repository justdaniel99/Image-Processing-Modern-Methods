import argparse
import cv2
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # mirror {h|v|c|d}
    mirror_parser = subparsers.add_parser("mirror")
    mirror_parser.add_argument("params", nargs=1)

    # extract (left_x) (top_y) (width) (height)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("params", nargs=4)

    # rotate {cw|ccw} (angle)
    rotate_parser = subparsers.add_parser("rotate")
    rotate_parser.add_argument("params", nargs=2)

    # autorotate
    subparsers.add_parser("autorotate")

    parser.add_argument("input_file", nargs=1)
    parser.add_argument("output_file", nargs=1)

    return parser


# image file reading
def image_file_read(image_file):
    return cv2.imread(image_file)


# image file saving
def image_file_save(image_file_path, image_file):
    cv2.imwrite(image_file_path, image_file)


# image (like a matrix) transposition
def image_matrix_transposition(image):
    transposed_image = np.zeros((len(image[0]), len(image), 3))  # image (like a matrix) transposition
    for i in range(len(image)):
        for j in range(len(image[0])):
            transposed_image[j][i] = image[i][j]
    return transposed_image


# mirror {h|v|c|d}
# image file mirroring along different axes
def mirror(params, input_image_file, output_image_file):

    if params[0] == "h":  # mirroring along a horizontal axis
        image_mirror = (image_file_read(image_file=input_image_file))[::-1, :, :]
    if params[0] == "v":  # mirroring along a vertical axis
        image_mirror = (image_file_read(image_file=input_image_file))[:, ::-1, :]
    if params[0] == "d":  # mirroring along a main diagonal axis
        image_mirror = image_matrix_transposition(image_file_read(image_file=input_image_file))
    if params[0] == "cd":  # mirroring along a secondary diagonal axis
        image_mirror = \
            ((image_matrix_transposition(image_file_read(image_file=input_image_file)))[::-1, :, :])[:, ::-1, :]

    image_file_save(output_image_file, image_mirror)  # saving mirrored image in the file


# rotating the image clockwise through angle
def image_rotate(image, angle):
    rotation_times = (angle // 90) % 4

    if rotation_times == 0:
        rotated_image = image

    for i in range(rotation_times):
        rotated_image = (image_matrix_transposition(image))[:, ::-1, :]
        image = rotated_image

    return rotated_image


# rotate {cw|ccw} (angle)
# rotating the image file
def rotate(params, input_image_file, output_image_file):  # работает на 90 градусов по и против

    params[1] = int(params[1])

    if params[0] == "ccw":
        params[1] = - params[1]

    image_file_save(output_image_file, image_rotate(image_file_read(image_file=input_image_file), params[1]))


# extract (left_x) (top_y) (width) (height)
# extracting the image part
def extract(params, input_image_file, output_image_file):
    image = image_file_read(image_file=input_image_file)  # image file reading

    params[0] = int(params[0])
    params[1] = int(params[1])
    params[2] = int(params[2])
    params[3] = int(params[3])

    extracted_image = np.zeros((params[3], params[2], 3))  # new image file with input width and height

    for i in range(0, params[3]):
        if ((i + params[1]) <= (len(image) - 1)) & ((i + params[1]) >= 0):
            for j in range(0, params[2]):
                if ((j + params[0]) <= (len(image[0]) - 1)) & ((j + params[0]) >= 0):
                    extracted_image[i][j] = image[i + params[1]][j + params[0]]  # considering x and y offset

    image_file_save(output_image_file, extracted_image)  # saving extracted image in the file


# autorotate
# automatic rotation
def autorotate(input_image_file, output_image_file):
    image = image_file_read(image_file=input_image_file)  # image file reading

    sum1 = 0  # sum of the pixels of the upper left part of the image
    sum2 = 0  # sum of the pixels of the lower left part of the image
    sum3 = 0  # sum of the pixels of the upper right part of the image
    sum4 = 0  # sum of the pixels of the lower right part of the image

    for i in range(0, len(image) // 2):
        for j in range(0, len(image[0]) // 2):
            sum1 = sum1 + image[i][j][0]

    for i in range(len(image) // 2, len(image)):
        for j in range(0, len(image[0]) // 2):
            sum2 = sum2 + image[i][j][0]

    for i in range(0, len(image) // 2):
        for j in range(len(image[0]) // 2, len(image[0])):
            sum3 = sum3 + image[i][j][0]

    for i in range(len(image) // 2, len(image)):
        for j in range(len(image[0]) // 2, len(image[0])):
            sum4 = sum4 + image[i][j][0]

    # if the light part of the picture is on the right
    if sum3 + sum4 == max(sum1 + sum2, sum1 + sum3, sum2 + sum4, sum3 + sum4):
        image_file_save(output_image_file, image_rotate(image, -90))

    # if the light part of the picture is at the top (as it should be)
    if sum1 + sum3 == max(sum1 + sum2, sum1 + sum3, sum2 + sum4, sum3 + sum4):
        image_file_save(output_image_file, image)

    # if the light part of the picture is on the left
    if sum1 + sum2 == max(sum1 + sum2, sum1 + sum3, sum2 + sum4, sum3 + sum4):
        image_file_save(output_image_file, image_rotate(image, 90))

    # if the light part of the picture is at the bottom
    if sum2 + sum4 == max(sum1 + sum2, sum1 + sum3, sum2 + sum4, sum3 + sum4):
        image_file_save(output_image_file, image_rotate(image, 180))


if __name__ == '__main__':

    namespace = create_parser().parse_args()

    if namespace.command == "mirror":
        mirror(namespace.params, namespace.input_file[0], namespace.output_file[0])
    elif namespace.command == "extract":
        extract(namespace.params, namespace.input_file[0], namespace.output_file[0])
    elif namespace.command == "rotate":
        rotate(namespace.params, namespace.input_file[0], namespace.output_file[0])
    else:
        autorotate(namespace.input_file[0], namespace.output_file[0])
