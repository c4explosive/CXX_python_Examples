import numpy as np
import cv2
import cxxBridge
import sys

def _drawCvHistogram(hist, zoom=1, color=(255,255,255)): # PURE OpenCV
        _, maxVal, _, _ = cv2.minMaxLoc(hist)
        histSize = hist.shape[0]

        img = np.empty((int(histSize*zoom), int(histSize*zoom), 3),
                        dtype=np.uint8)
        img[:, :] = 0
        
        hpt = int(0.9*histSize)

        for h in range(0, histSize):
            binVal = hist[h]

            if (binVal>0):
                intensity = int(binVal*hpt/maxVal)
                cv2.line(img, (int(h*zoom), int(histSize*zoom)),
                            (int(h*zoom), int((histSize-intensity)*zoom)), 
                            color, zoom)

        return img



def main():
    img = cv2.imread(sys.argv[1])
    img = cv2.convertScaleAbs(img, img, 1, 0)
    hists, maxs, mat33 = cxxBridge.numpy_bridge(img)
    histimgb = _drawCvHistogram(hists[0], color=(255,0,0))
    histimgg = _drawCvHistogram(hists[1], color=(0,255,0))
    histimgr = _drawCvHistogram(hists[2], color=(0,0,255))

    imgbg = cv2.resize(img, histimgb.shape[:2])
    imgs = cv2.resize(img, ( int(histimgb.shape[0] * (img.shape[1]/img.shape[0]) ), histimgb.shape[0]))
    #imgbg[:,:] = (0,0,0)

    offset = int(imgbg.shape[1]/2 - imgs.shape[1]/2)

    ratio = img.shape[1]/img.shape[0]
    '''
    if(ratio >= 1.0):
        imgbg[:,0:imgs.shape[1]] = imgs[:,0:imgbg.shape[1]]
    else:
        imgbg[:,0+offset:imgs.shape[1]+offset] = imgs
    '''

    histimg = cv2.addWeighted(imgbg, 1, histimgb, 0.5, 0)
    histimg = cv2.addWeighted(histimg, 0.5, histimgg, 0.5, 0)
    histimg = cv2.addWeighted(histimg, 0.5, histimgr, 0.5, 0)

    print(f"Mat33:: {mat33}")

    cv2.namedWindow("+s", cv2.WINDOW_NORMAL)
    cv2.imshow("+s", histimg)
    #cv2.imshow("+s", mat33)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()