import cv2
import numpy as np
import math

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)

    # test code
    cv2.imshow("OpenCV Test", image)
    out=GaussianFilter_JHAn(image)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    dap= cv2.GaussianBlur(image, (0,0), 3)
    cv2.imshow("dap",dap)
    cv2.imshow("GaussianFilter_JHAn",out_u8)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter


# implement your code here

def GaussianFilter_JHAn(image: np.ndarray) -> np.ndarray:
    print(image.shape)
    out=np.zeros(image.shape)

    img=image.astype(np.float32)

    sigma_d=3
    r=math.ceil(3*sigma_d)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            pixel_value=img[i,j]
            dx=0
            result=0
            div=0
            
            c=0
            for k in range(i-r,i+r+1):
                for l in range(j-r,j+r+1):
                    if k<0 or k>=img.shape[0] or l<0 or l>=img.shape[1]:
                        continue
                    dx=((i-k)**2+(j-l)**2)
                    c=np.exp((-1/2)*(dx)/(sigma_d**2))
                    div+=c

                    result+=c*img[k,l]
            out[i,j]=(result/div)
    
    return out



if __name__ == "__main__":
    main()