import cv2
import numpy as np

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    GaussianFilter_JWChoi(image)
    GaussianFilter_JHAn(image)
    GaussianFilter_EKBu(image)


# implement your code here
def GaussianFilter_JWChoi(image: np.ndarray) -> np.ndarray:
    # to do

def GaussianFilter_JHAn(image: np.ndarray) -> np.ndarray:
    # to do

def GaussianFilter_EKBu(image: np.ndarray) -> np.ndarray:
    # to do

if __name__ == "__main__":
    main()