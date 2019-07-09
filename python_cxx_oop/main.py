import cxxOop #Acts like python module but is a CXX code

def main():
    pt1 = (14,2)
    pt2 = (7,4)
    print(f"Euclidean dist: {cxxOop.euclidean_dist(*pt1,*pt2)}.")
    pt2 = (2,1)
    print(f"Euclidean dist mod: {cxxOop.euclidean_dist(*pt1,*pt2)}.")

if __name__ == "__main__":
    main()