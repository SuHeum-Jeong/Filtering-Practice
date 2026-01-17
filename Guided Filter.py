import cv2
import numpy as np

def main():
    image = cv2.imread("image/cameraman.png", cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    # test code
    out=GuidedFilter_JHAn(image)
    cv2.imshow("OpenCV Test", image)
    cv2.imshow("GuidedFilter_JHAn", out)

    I = image.astype(np.float32) / 255.0

    # Guided Filter
    r = 8
    eps = 1e-3
    out = cv2.ximgproc.guidedFilter(
        guide=I,
        src=I,
        radius=r,
        eps=eps
    )
    cv2.imshow("dap", (out*255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Gaussian Filter
    


# implement your code here

def add_pad(image,r):
    h=image.shape[0]
    w=image.shape[1]
    hh=h+2*r;ww=w+2*r
    out=[[0]*ww for _ in range(hh)]
    def clamp(v,lo,hi):
        return lo if v<lo else hi if v>hi else v
    for y in range(hh):
        for x in range(ww):
            yy=clamp(y-r,0,h-1)
            xx=clamp(x-r,0,w-1)
            out[y][x]=image[yy][xx]
    return out
def accumulate_sum(image):
    h=image.shape[0]
    w=image.shape[1]
    out=[[0]*(w+1) for _ in range(h+1)]
    for y in range(h):
        row_sum=0
        for x in range(w):
            row_sum+=image[y][x]
            out[y+1][x+1]=out[y][x+1]+row_sum

    return out

def get_mean(image,r):
    h=image.shape[0]
    w=image.shape[1]

    p=add_pad(image,r)
    p = np.array(add_pad(image, r), dtype=np.float32)
    s=accumulate_sum(p)
    
    ww=2*r+1
    area=ww*ww
    out = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        yp=y+r;y1=yp-r;y2=yp+r
        y1i,y2i=y1,y2+1
        for x in range(w):
            xp=x+r;x1=xp-r;x2=xp+r
            x1i,x2i=x1,x2+1

            total=s[y2i][x2i]-s[y1i][x2i]-s[y2i][x1i]+s[y1i][x1i]
            out[y][x]=total/area

    return out

def GuidedFilter_JHAn(image: np.ndarray) -> np.ndarray:
    
    image= image.astype(np.float64)
    mean_I=get_mean(image,8)
    mean_p=get_mean(image,8)
    mean_II=get_mean(image*image,8)
    mean_IP=get_mean(image*image,8)

    var_I=mean_II-mean_I*mean_I
    cov_Ip=mean_IP-mean_I*mean_p

    a=cov_Ip/(var_I+0.00001)
    b=mean_p-a*mean_I

    mean_a=get_mean(a,8)
    mean_b=get_mean(b,8)

    q=mean_a*image+mean_b
    q = np.clip(q, 0, 255).astype(np.uint8)

    return q
    
if __name__ == "__main__":
    main()