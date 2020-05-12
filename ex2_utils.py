from collections import defaultdict
from math import pi, cos, sin

import cv2
import numpy as np


# from canny import canny_edge_detector


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207950577


sobel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 8.0
sobel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 8.0


# 1

def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    # signal_rev = np.flipud(inSignal)
    # signal_len = signal_rev.shape[0]
    # kernel_len = kernel1.shape[0]
    # l = kernel_len // 2
    # signal_conv = np.zeros(signal_len)
    #
    # for j in range(l, signal_len - l):
    #     sum = 0
    #     for n in range(kernel_len):
    #         sum = sum + kernel1[n] * signal_rev[j - l + n]
    #     signal_conv[j] = sum
    # return signal_conv
    return conv2D(np.array([inSignal]), np.array([kernel1]))


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    img_h, img_w = inImage.shape
    ker_h, ker_w = kernel2.shape
    pad = (ker_w - 1) // 2  # "pad" is the borders of the input image
    print(pad)
    image = cv2.copyMakeBorder(inImage, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    print(image)
    output = np.zeros((img_h, img_w))
    for y in range(pad, img_h + pad):
        for x in range(pad, img_w + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            output[y - pad, x - pad] = (roi * kernel2).sum()
    # output = rescale_intensity(output, in_range=(0, 255))
    # output = (output * 255).astype("uint8")
    # output = (output * 255)
    return output


# 2

def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude,x_der,y_der)
    """
    Gx = np.array([[-1, 0, 1]])
    Gy = Gx.transpose()

    x_der = conv2D(inImage, Gx)
    y_der = conv2D(inImage, Gy)

    directions = np.rad2deg(np.arctan2(y_der, x_der))
    directions[directions < 0] += 180

    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    magnitude = (magnitude / np.max(magnitude)) * 255

    return directions, magnitude, x_der, y_der


# bonus

def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    return conv2D(in_image, kernel_size)


def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    return cv2.filter2D(in_image, -1, kernel_size, borderType=cv2.BORDER_REPLICATE)


# bonus


# 3

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    x_der = conv2D(img, sobel_x)
    y_der = conv2D(img, sobel_y)
    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    sob_result = (magnitude / np.max(magnitude)) * 255
    sob_result[sob_result < thresh * 255] = 0
    # using cv2:
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    combine = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return combine, sob_result


# one of the two

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    img = cv2.GaussianBlur(img, (0, 0), 1) if 0. < 1 else img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows - 2 + r, c:cols - 2 + c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows - 1, 1:cols - 1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    log_img = values.astype(np.uint8)
    return log_img


# def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
#     """
#     Detecting edges using the "ZeroCrossingLOG" method
#     :param img: Input image
#     :return: :return: Edge matrix
#     """


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    img = cv2.GaussianBlur(img, (5, 5), 0)
    directions, magnitude = convDerivative(img)[:2]
    result = non_maximum_suppression(magnitude, directions)
    result = double_threshold_hysteresis(result, thrs_1, thrs_2)

    # using cv2:
    edges_cv2 = cv2.Canny(img, thrs_1, thrs_2)

    return edges_cv2, result


def non_maximum_suppression(image, angles):
    suppressed = np.zeros(image.shape)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif 22.5 <= angles[i, j] < 67.5:
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif 67.5 <= angles[i, j] < 112.5:
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    # suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    out = np.zeros(image.shape, dtype=np.uint8)
    strong_i, strong_j = np.where(image >= high)
    zeros_i, zeros_j = np.where(image < low)

    # weak edges
    weak_i, weak_j = np.where((image <= high) & (image >= low))

    # Set same intensity value for all edge pixels
    out[strong_i, strong_j] = 255
    out[zeros_i, zeros_j] = 0
    out[weak_i, weak_j] = 75
    M, N = out.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if out[i, j] == 75:
                if 255 in [out[i + 1, j - 1], out[i + 1, j], out[i + 1, j + 1], out[i, j - 1], out[i, j + 1],
                           out[i - 1, j - 1], out[i - 1, j], out[i - 1, j + 1]]:
                    out[i, j] = 255
                else:
                    out[i, j] = 0
    return out


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    bimg = cv2.bilateralFilter(img, 5, 175, 175)

    cann = cv2.Canny(bimg, 100, 200)

    pixel = np.argwhere(cann == 255)

    accum = [[[0 for r in range(int(min_radius), int(max_radius))] for h in range(0, 30)] for k in range(0, 30)]
    print(accum)
    for r in range(int(min_radius), int(max_radius)):
        for h in range(0, 30):
            for k in range(0, 30):
                for p in pixel:
                    # print r,h,k,p
                    xpart = (h - p[0]) ** 2
                    ypart = (k - p[1]) ** 2
                    rhs = xpart + ypart
                    lhs = r * r
                    if (lhs == rhs):
                        accum[k][h][r - int(min_radius)] += 1

    print(accum)
    return accum
