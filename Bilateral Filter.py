import cv2
import numpy as np

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    BilateralFilter_JWChoi(image)
    BilateralFilter_JHAn(image)
    BilateralFilter_EKBu(image)
    
    return


# implement your code here
def BilateralFilter_JWChoi(image: np.ndarray) -> np.ndarray:
    # to do
    return

def BilateralFilter_JHAn(image: np.ndarray) -> np.ndarray:
    # to do
    return

def BilateralFilter_EKBu(image: np.ndarray) -> np.ndarray:
    def gaussian_function(x, y, sigma):
        coefficient = 1 / (2 * np.pi * (sigma**2))
        exponent = -(x**2 + y**2) / (2 * (sigma**2))

        return coefficient * np.exp(exponent)
            


    height = image.shape[0]
    width = image.shape[1]
    new_img = np.zeros((height, width), dtype = np.float32)

    k = 5
    for y in range(height):
        for x in range(width):
            weight = 0.0
            weighted_sum = 0.0
            sum_of_weight = 0.0
            for j in range(max(0,y - k//2),min(height, y+k//2 + 1)):
                for i in range(max(0, x - k//2), min(width, x+k//2 + 1)):
                    close = gaussian_function(i-x,j-y,k/6)
                    pixel_diff = float(image[y,x]) - float(image[j,i])
                    similarity = gaussian_function(pixel_diff , 0 , 10)
                    weight = close*similarity
                    weighted_sum += weight*image[j,i]
                    sum_of_weight += weight
            if sum_of_weight > 0:
                new_img[y,x] = weighted_sum / sum_of_weight
    new_img = new_img.astype(np.uint8)
    cv2.imshow("OpenCV Test", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 0
if __name__ == "__main__":
    main()