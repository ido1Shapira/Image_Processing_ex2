import time

from ex2.ex2_utils import *
import cv2
import matplotlib.pyplot as plt

thresh_dict = {
    "beach.jpg": (0.6, 0.4, 0.7),  # (soble_thresh, canny_thresh1, canny_thresh2)
    "boxman.jpg": (0.4, 0.3, 0.45),  # (soble_thresh, canny_thresh1, canny_thresh2)
    "cloun.jpeg": (0.7, 0.5, 0.7),  # (soble_thresh, canny_thresh1, canny_thresh2)
    "small-circles.jpg": (10, 15),  # (min_radios, max_radios)
    "circles.jpg": (50, 60),  # (min_radios, max_radios)
}


def test_conv2d(img: np.ndarray):
    kernel = np.ones((5, 5))
    kernel /= kernel.sum()
    cv2_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    st = time.time()
    conv_img = conv2D(img, kernel)
    print("Time:%.2f" % (time.time() - st))
    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(cv2_img)
    ax[0].set_title("conv2d cv2")
    ax[1].imshow(conv_img)
    ax[1].set_title("my implementation")
    plt.show()


def test_convDerivative(img: np.ndarray):
    st = time.time()
    directions, magnitude, x_der, y_der = convDerivative(img)
    print("Time:%.2f" % (time.time() - st))
    plt.title("convDerivative")
    f, ax = plt.subplots(2, 2)
    plt.gray()
    ax[0][0].imshow(directions)
    ax[0][0].set_title("directions")
    ax[0][1].imshow(magnitude)
    ax[0][1].set_title("magnitude")
    ax[1][0].imshow(x_der)
    ax[1][0].set_title("x_der")
    ax[1][1].imshow(y_der)
    ax[1][1].set_title("y_der")
    plt.show()

def test_blurImage12(img: np.ndarray, kernel_size: int):
    st = time.time()
    blur_img = blurImage1(img, kernel_size)
    print("Time:%.2f" % (time.time() - st))
    st = time.time()
    cv2_blur = blurImage2(img, kernel_size)
    print("Time:%.2f" % (time.time() - st))

    f, ax = plt.subplots(1, 2)
    plt.gray()
    ax[0].imshow(cv2_blur)
    ax[0].set_title("blurImage2")
    ax[1].imshow(blur_img)
    ax[1].set_title("my implementation")
    plt.show()

def test_edgeDetectionSobel(img: np.ndarray, thresh: float = 0.6):
    st = time.time()
    cv2_edge_img, edge_img = edgeDetectionSobel(img, thresh)
    print("Time:%.2f" % (time.time() - st))

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2_edge_img)
    ax[0].set_title("Sobel cv2")
    ax[1].imshow(edge_img)
    ax[1].set_title("my implementation")
    plt.show()


def test_edgeDetectionZeroCrossingSimple(img: np.ndarray):
    st = time.time()
    edge_img = edgeDetectionZeroCrossingSimple(img)
    print("Time:%.2f" % (time.time() - st))

    plt.imshow(edge_img)
    plt.title("edgeDetectionZeroCrossingSimple")
    plt.show()


def test_edgeDetectionZeroCrossingLOG(img: np.ndarray):
    st = time.time()
    edge_img = edgeDetectionZeroCrossingLOG(img)
    print("Time:%.2f" % (time.time() - st))

    plt.imshow(edge_img)
    plt.title("edgeDetectionZeroCrossingLOG")
    plt.show()


def test_edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float):
    st = time.time()
    cv2_edge_img, edge_img = edgeDetectionCanny(img, thrs_1, thrs_2)
    print("Time:%.2f" % (time.time() - st))

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2_edge_img)
    ax[0].set_title("Canny cv2")
    ax[1].imshow(edge_img)
    ax[1].set_title("my implementation")
    plt.show()


def test_houghCircle(img: np.ndarray, min_radius: float, max_radius: float):
    st = time.time()
    res = houghCircle(img, min_radius, max_radius)
    print("Time:%.2f" % (time.time() - st))

    fig = plt.figure()
    plt.imshow(img)
    circle = []
    for r, x, y in res:
        circle.append(plt.Circle((y, x), r, color=(1, 0, 0), fill=False))
        fig.add_subplot().add_artist(circle[-1])
    plt.title("houghCircle")
    plt.show()
    # for x, y, r in res:
    #     cv2.circle(img1, (y, x), r, (255, 255, 255) ,)
    # plt.imshow(img1)
    # plt.show()


def main():
    print("ID:", myID())
    plt.gray()

    a = [1, 2, 3, 4]
    k = [0, 1, 0]
    print('conv: ', np.convolve(a, k, 'full'))
    print('conv1D: ', conv1D(np.array(a), np.array(k)))

    for img_path in thresh_dict.keys():
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print('image shape: ', img.shape)
        if img_path == 'small-circles.jpg' or img_path == 'circles.jpg':
            print('Run houghCircle with min,max= ', thresh_dict[img_path][0], thresh_dict[img_path][1])
            test_houghCircle(img, thresh_dict[img_path][0], thresh_dict[img_path][1])
            continue
        print('Runs all functions of the task on the image: "', img_path, '"')
        test_conv2d(img)
        test_convDerivative(img)
        test_blurImage12(img, 5)
        print('Run Sobel with thresh= ', thresh_dict[img_path][0])
        test_edgeDetectionSobel(img, thresh_dict[img_path][0])
        test_edgeDetectionZeroCrossingSimple(img)
        test_edgeDetectionZeroCrossingLOG(img)
        print('Run Canny with thresh1,2 = ', thresh_dict[img_path][1], thresh_dict[img_path][2])
        test_edgeDetectionCanny(img, thresh_dict[img_path][1], thresh_dict[img_path][2])


if __name__ == '__main__':
    main()
