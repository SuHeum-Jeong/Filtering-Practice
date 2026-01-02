import cv2
import numpy as np
import math

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    #test code
    cv2.imshow("OpenCV Test", image)

    #Gaussian Filter
    out=BilateralFilter_JHAn(image)
    dap=cv2.bilateralFilter(image, d=19, sigmaColor=10, sigmaSpace=3)
    cv2.imshow("OpenCV Bilateral Filter", dap)

    cv2.imshow("Bilateral Filter JHAn", out.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# implement your code here

def BilateralFilter_JHAn(image: np.ndarray) -> np.ndarray:
    # to do
    print("BilateralFilter_JHAn is not implemented yet.")
    print(image.shape)
    out=np.zeros(image.shape)
    img=image.astype(np.float32)
    
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
            
    #         pixel_value=image[i,j]
    #         dx=0
    #         scale=360
    #         result=0
    #         div=0
    #         s=0
    #         c=0
            
    #         for k in range(image.shape[0]):
    #             for l in range(image.shape[1]):
    #                 dx=math.sqrt((i-k)**2+(j-l)**2)
    #                 s=np.exp((-1/2)*math.fabs(pixel_value-image[k,l])**2/(scale**2))
    #                 c=np.exp((-1/2)*(dx**2)/(scale**2))
    #                 div+=c*s

    #                 result+=c*s*image[k,l]
    #         out[i,j]=(result/div)

    #-> 1개의 픽셀에 대해 처리 해줄 때 256x256 전체 이미지를 다 돌아야함-> 너무 느림
    #-> 주변 픽셀들만 돌도록 수정 필요

    sigma_d=3
    sigma_s=10
    r=math.ceil(3*sigma_d)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            pixel_value=img[i,j]
            dx=0
            
            result=0
            div=0
            s=0
            c=0
            for k in range(i-r,i+r+1):
                for l in range(j-r,j+r+1):
                    if k<0 or k>=img.shape[0] or l<0 or l>=img.shape[1]:
                        continue
                    dx=((i-k)**2+(j-l)**2)
                    s=np.exp((-1/2)*(pixel_value-img[k,l])**2/(sigma_s**2))
                    c=np.exp((-1/2)*(dx)/(sigma_d**2))
                    div+=c*s

                    result+=c*s*img[k,l]
            out[i,j]=(result/div)
    
    return out

if __name__ == "__main__":
    main()