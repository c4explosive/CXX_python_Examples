import sys
import cv2
import cxxDnn
import numpy as np
import json

def see_results(tensor_detects, img, sjson):
    for ten_dect in tensor_detects:
        coords = np.int32(ten_dect[0:4])
        if (ten_dect[4] > 84): ten_dect[4] = 1
        tclass, score = sjson["key"][int(ten_dect[4]-1)], ten_dect[5]
        #print(f"Tensor_Detect: {coords}")
        cv2.rectangle(img, (coords[0],coords[1]), 
                    (coords[2], coords[3]), (0, 255, 0), 2)
        cv2.putText(img, f"{tclass}", 
                    (coords[2], coords[1]+20), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255,0,0), 2)
        score = round(score*100, 2)
        cv2.putText(img, str(score)+"%", 
                    (coords[2], coords[1]+50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255,0,0), 2)
        print(f"Data: {str(round(score, 2))}% || {tclass}")
    cv2.namedWindow("+s", cv2.WINDOW_NORMAL)
    cv2.imshow("+s", img)
    cv2.waitKey(0)


def detect(filename):
    sjson = json.loads(open("mscoco.json").read())
    img = cv2.imread(filename)
    if (img is None): return None, None, []
    tensor_detects = cxxDnn.dnn_bridge(img)
    return img, sjson, tensor_detects


def main():
    filename = sys.argv[1]
    img, sjson, tensor_detects = detect(filename) 

    print(tensor_detects)

    SEE_RES = False

    if(SEE_RES):
        see_results(tensor_detects, img, sjson)

    

if __name__ == "__main__":
    main()