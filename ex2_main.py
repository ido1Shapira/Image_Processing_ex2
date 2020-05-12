from ex2.ex2_utils import *
import cv2
import matplotlib.pyplot as plt

def main():
    print("ID:", myID())
    img_path = 'beach.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.gray()
    plt.imshow(img)
    plt.show()

    # 1

    a = [1, 2, 3, 4]
    k = [0, 1, 0]
    print(np.convolve(a, k, 'full'))
    print(conv1D(np.array(a), np.array(k)))
    print(np.array([a]).shape)

    # kernel = np.ones((9, 9))
    # kernel /= kernel.sum()
    # cv2_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # conv_img = conv2D(img, kernel)
    # plt.imshow(conv_img)
    # plt.show()
    # plt.imshow(cv2_img)
    # plt.show()

    # edge_img = edgeDetectionZeroCrossingSimple(img)
    # plt.imshow(edge_img)
    # plt.show()

    cv2_edge_img, edge_img = edgeDetectionSobel(img, 0)
    plt.imshow(edge_img)
    plt.show()
    plt.imshow(cv2_edge_img)
    plt.show()

    cv2_edge_img, edge_img = edgeDetectionCanny(img, 50, 110)
    plt.imshow(edge_img)
    plt.show()
    plt.imshow(cv2_edge_img)
    plt.show()

    # img_path = 'circle.png'
    # img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img1)
    # plt.show()
    #
    # img_circles = houghCircle(img1, 18, 20)
    # for x, y, r in img_circles:
    #     cv2.circle(img1, (x, y), r, (255, 0, 0))
    # plt.imshow(img1)
    # plt.show()

if __name__ == '__main__':
    main()
