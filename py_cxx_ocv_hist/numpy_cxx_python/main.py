import numpy as np
import cv2
import cxxBridge
import sys

def main():
    array = np.array([120, 5, 6, 8, 9], dtype=np.uint8)
    multiarray = np.array([[9, 8, 7, 4, 3], [6, 3, 8, 7, 1]], dtype=np.uint8)
    multiarray = cv2.imread(sys.argv[1])
    print(f"Array: {array}")
    print(f"MultiArray: {multiarray}")
    print(array.dtype)
    print("Python_addr:: ", hex(id(array)))
    nnm = cxxBridge.numpy_bridge(array, multiarray)
    print(f"mArray_vect: {nnm}")

if __name__ == "__main__":
    main()