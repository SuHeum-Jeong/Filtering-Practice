import cv2
import numpy as np

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    GuidedFilter_EKBu(image)

    return 

def accumulate_sum(image):
    height, width = image.shape
    sum_table = np.zeros((height + 1, width + 1), dtype=np.float32)
    
    img_float = image.astype(np.float32)

    for y in range(1, height + 1):
        for x in range(1, width + 1):

            sum_table[y, x] = img_float[y-1, x-1] + sum_table[y-1, x] + sum_table[y, x-1] - sum_table[y-1, x-1]

    return sum_table



def get_mean(image, r):
    height = image.shape[0]
    width = image.shape[1]
    S = np.zeros((height, width), dtype=np.float32)
    S = accumulate_sum(image)

    result = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            y1, y2 = max(0, y - r), min(height - 1, y + r)
            x1, x2 = max(0, x - r), min(width - 1, x + r)

            rect_sum = S[y2+1, x2+1] - S[y1, x2+1] - S[y2+1, x1] + S[y1, x1]
            
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            result[y, x] = rect_sum / area
            
    return result

                    
def GuidedFilter_EKBu(image):

    I = image.astype(np.float32) / 255.0
    p = I.copy()
    
    r = 4
    eps = 0.01

    # 평균 계산
    mean_I = get_mean(I, r)
    mean_p = get_mean(p, r)
    mean_II = get_mean(I * I, r)
    mean_Ip = get_mean(I * p, r)

    # 분산 및 공분산 계산
    var_I = mean_II - (mean_I * mean_I)
    cov_Ip = mean_Ip - (mean_I * mean_p)

    # 선형 계수 a, b 계산
    a = cov_Ip / (var_I + eps)
    b = mean_p - (a * mean_I)

    # 계수 a, b의 평균 계산
    mean_a = get_mean(a, r)
    mean_b = get_mean(b, r)

    # 최종 출력 이미지 조립
    q = mean_a * I + mean_b

    # 0~255 범위로 다시 돌려주고 uint8로 변환하여 반환
    q = (q * 255.0).clip(0, 255).astype(np.uint8)
    
    cv2.imshow("Guided Filter Result", q)
    cv2.waitKey(0)
    
    return q

if __name__ == "__main__":
    main()






