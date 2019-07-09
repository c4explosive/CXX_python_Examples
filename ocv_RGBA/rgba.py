import cv2
import sys
import numpy as np
import cxxRGBA

def prepareMask(imgpA):
    if(len(imgpA.shape) >= 3):
        if(imgpA.shape[2] == 2):
            tmp = np.zeros((*imgpA.shape[0:2], 3), imgpA.dtype)
            imgpA[:,:,::-1]
            tmp[:,:,0:2] = imgpA[:,:,0:2]
            #tmp[:,:,1] = imgpA[:,:,0]
            imgpA = tmp.copy()
            imgpA = cv2.cvtColor(imgpA, cv2.COLOR_BGR2GRAY)
            
        if(imgpA.shape[2] == 4): # Have alpha
            imgpA = imgpA.astype(np.float)
            #print(imgpA[0,0])
            alpha = imgpA[:, :, 3] / 255. # Alpha intensity of all pixels (This is weighted 0-1)
            imgRgb = imgpA[:, :, 0:3]
            alpha = alpha * 0.5
            alpha = alpha[:,:,np.newaxis]
            imgRgb = imgRgb * alpha
            alpha = np.clip((alpha)*255, 0, 255)
            imgpA = np.clip(imgRgb+alpha, 0, 255)
            imgpA = imgpA.astype(np.uint8)
            #print("Alpha_shape:: ", imgpA.shape)
        
        imgpA = cv2.cvtColor(imgpA, cv2.COLOR_BGR2GRAY)

        return imgpA
    else:
        return imgpA

def main():
    base = 800

    img = cv2.imread(sys.argv[2])
    imgpA = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    #imgpA[imgpA == 255] = 0
    imgpA = imgpA.astype(np.uint8)
    #imgpA = cv2.bitwise_not(imgpA) # Alpha mask become binary_Mask
    if(len(sys.argv) >= 4):
        imgB = cv2.imread(sys.argv[3])
        M = cv2.getRotationMatrix2D((int(imgB.shape[1]/2), int(imgB.shape[0]/2)), -90, 1)
        #imgB = cv2.warpAffine(imgB, M, (imgB.shape[0], imgB.shape[1]))
        imgB = cv2.resize(imgB, (base, base))
    else:
        imgB = None

    if(len(sys.argv) >= 6):
        imgC = cv2.imread(sys.argv[4])
        imgC = cv2.resize(imgC, (base, base))
        imgM2 = cv2.imread(sys.argv[5], cv2.IMREAD_UNCHANGED)
        #print(imgM2.shape)
        imgM2 = prepareMask(imgM2)
        imgM2 = cv2.resize(imgM2, (base, base))
    else:
        imgC = None
        imgM2 = None

    #print(imgpA.shape)


    imgpA = prepareMask(imgpA)


    imgpA = cv2.resize(imgpA, (base, base))
    img = cv2.resize(img, (base, base))

    img = cxxRGBA.alpha_process(imgpA, img, imgB)

    if(imgC is not None):
        img = cxxRGBA.alpha_process(imgM2, img, imgC)

    if(img is not None):
        cv2.namedWindow("+s", cv2.WINDOW_NORMAL)
        cv2.imshow("+s", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()