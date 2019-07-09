import cxxOpencv
import sys

def main():
    y, x = cxxOpencv.mat_see_data(sys.argv[1])
    print(f"x: {x}, y: {y}.")

if __name__ == "__main__":
    main()