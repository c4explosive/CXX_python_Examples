void kernel alpha_blending(global const float* A, global const float* B, global float* C, global float* D,
                                  global const int* N) 
{
    int ID, Nthreads, n, ratio, start, stop;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    n = N[0];

    ratio = (n / Nthreads);  // number of elements for each thread
    start = ratio * ID;
    stop  = ratio * (ID + 1);

    for (int i=start; i<stop; i++) // A -> alpha, B-> bg, D-> fg, C -> out
        *(C+i) =  *(A+i) * *(D+i) + (1-*(A+i)) * *(B+i); // out = alpha*fg + (1-alpha)*bg
}