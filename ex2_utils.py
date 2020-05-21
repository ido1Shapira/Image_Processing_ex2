import cv2
import numpy as np

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207950577


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
laplacian_ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


# 1

def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    k_len = len(kernel1)
    inSignal = np.pad(inSignal, (k_len - 1, k_len - 1), 'constant')
    sig_len = len(inSignal)
    signal_conv = np.zeros(sig_len - k_len + 1)
    for i in range(sig_len - k_len + 1):
        signal_conv[i] = (inSignal[i:i + k_len] * kernel1).sum()
    return signal_conv


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    kernel2 = np.flip(kernel2)
    img_h, img_w = inImage.shape
    ker_h, ker_w = kernel2.shape
    image_padded = np.pad(inImage, (ker_h // 2, ker_w // 2), 'edge')
    output = np.zeros((img_h, img_w))
    for y in range(img_h):
        for x in range(img_w):
            output[y, x] = (image_padded[y:y + ker_h, x:x + ker_w] * kernel2).sum()
    return output


# 2

def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale image
    :return: (directions, magnitude,x_der,y_der)
    """
    Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Gy = Gx.transpose()

    x_der = conv2D(inImage, Gx)
    y_der = conv2D(inImage, Gy)

    directions = np.rad2deg(np.arctan2(y_der, x_der))
    # directions[directions < 0] += 180

    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    # magnitude = magnitude * 255.0 / magnitude.max()

    return directions, magnitude, x_der, y_der


# bonus

def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    assert (kernel_size % 2 == 1)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    return conv2D(in_image, create_gaussian(kernel_size, sigma))


def create_gaussian(size, sigma):
    mid = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x, y = i - mid, j - mid
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return kernel


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    assert (kernel_size % 2 == 1)
    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

# bonus

# 3

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    assert (1 >= thresh >= 0)
    x_der = conv2D(img, sobel_x)
    y_der = conv2D(img, sobel_y)
    magnitude = np.sqrt(np.square(x_der) + np.square(y_der))
    magnitude[magnitude < thresh * 255] = 0
    magnitude[magnitude >= thresh * 255] = 1

    # using cv2:
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    combine = cv2.magnitude(grad_x, grad_y)
    combine[combine < thresh * 255] = 0
    combine[combine >= thresh * 255] = 1
    return combine, magnitude


# one of the two

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    img = conv2D(img, laplacian_ker)
    zero_crossing = np.zeros(img.shape)
    for i in range(img.shape[0] - (laplacian_ker.shape[0] - 1)):
        for j in range(img.shape[1] - (laplacian_ker.shape[1] - 1)):
            if img[i][j] == 0:
                if (img[i][j - 1] < 0 and img[i][j + 1] > 0) or \
                        (img[i][j - 1] < 0 and img[i][j + 1] < 0) or \
                        (img[i - 1][j] < 0 and img[i + 1][j] > 0) or \
                        (img[i - 1][j] > 0 and img[i + 1][j] < 0):  # All his neighbors
                    zero_crossing[i][j] = 255
            if img[i][j] < 0:
                if (img[i][j - 1] > 0) or (img[i][j + 1] > 0) or (img[i - 1][j] > 0) or (img[i + 1][j] > 0):
                    zero_crossing[i][j] = 255
    return zero_crossing


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param img: Input image
    :return: :return: Edge matrix
    """
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(img)


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges using "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    assert (1 >= thrs_1 >= 0)
    assert (1 >= thrs_2 >= 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    grad_x = conv2D(img, sobel_x)
    grad_y = conv2D(img, sobel_y)
    magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    directions = round_angle(np.rad2deg(np.arctan2(grad_y, grad_x)) % 180)
    result = non_maximum_suppression(magnitude, directions)
    result = double_threshold_hysteresis(result, thrs_1 * 255, thrs_2 * 255)

    # using cv2:
    edges_cv2 = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), thrs_1 * 255, thrs_2 * 255)

    return edges_cv2, result


def round_angle(angle: np.ndarray):
    angle[(angle < 22.5) | (157.5 <= angle)] = 0
    angle[(22.5 <= angle) & (angle < 67.5)] = 45
    angle[(67.5 <= angle) & (angle < 112.5)] = 90
    angle[(112.5 <= angle) & (angle < 157.5)] = 135
    return angle

def non_maximum_suppression(magnitude, Theta):
    ans = np.zeros(magnitude.shape)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if Theta[i, j] == 0:
                if (magnitude[i, j] > magnitude[i, j - 1]) and (magnitude[i, j] > magnitude[i, j + 1]):
                    ans[i, j] = magnitude[i, j]
            elif Theta[i, j] == 45:
                if (magnitude[i, j] > magnitude[i - 1, j + 1]) and (magnitude[i, j] > magnitude[i + 1, j - 1]):
                    ans[i, j] = magnitude[i, j]
            elif Theta[i, j] == 90:
                if (magnitude[i, j] > magnitude[i - 1, j]) and (magnitude[i, j] > magnitude[i + 1, j]):
                    ans[i, j] = magnitude[i, j]
            elif Theta[i, j] == 135:
                if (magnitude[i, j] > magnitude[i - 1, j - 1]) and (magnitude[i, j] > magnitude[i + 1, j + 1]):
                    ans[i, j] = magnitude[i, j]
    return ans

def All_his_neighbors(img, x, y):
    return [img[x - 1, y - 1], img[x - 1, y],
            img[x - 1, y + 1], img[x, y - 1],
            img[x, y + 1], img[x + 1, y - 1],
            img[x + 1, y], img[x + 1, y + 1]]

def double_threshold_hysteresis(img, low, high):
    img_h, img_w = img.shape
    result = np.zeros((img_h, img_w))
    result[img >= high] = 255
    weak_x_y = np.argwhere((img <= high) & (img >= low))
    for x, y in weak_x_y:
        result[x, y] = 255 if 255 in All_his_neighbors(result, x, y) else 0
    result[img < low] = 0
    return result

def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    img_h, img_w = img.shape
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 100)
    x_y_edges = np.argwhere(img > 0)
    A = np.zeros((max_radius, img_h + 2 * max_radius, img_w + 2 * max_radius))
    theta = np.arange(0, 360) * np.pi / 180
    for r in range(round(min_radius), round(max_radius)):
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r+1), 2 * (r+1)))
        (x_0, y_0) = (r+1, r+1)  # the center of the blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[x_0 + x, y_0 + y] = 1
        constant = np.argwhere(bprint).shape[0]

        for x, y in x_y_edges:  # For each edge coordinates
            A[r, x - x_0 + max_radius:x + x_0 + max_radius, y - y_0 + max_radius:y + y_0 + max_radius] += bprint
        threshold = 7
        A[r][A[r] < threshold * constant / r] = 0  # threshold

    # Extracting the circle information
    B = np.zeros((max_radius, img_h + 2 * max_radius, img_w + 2 * max_radius))
    region = 15  # Size to detect peaks
    for r, x, y in np.argwhere(A):
        environment = A[r - region:r + region, x - region:x + region, y - region:y + region]
        p, a, b = np.unravel_index(np.argmax(environment), environment.shape)
        B[r + (p - region), x + (a - region), y + (b - region)] = 1
    circleCoordinates = np.argwhere(B[:, max_radius:-max_radius, max_radius:-max_radius])
    return circleCoordinates
