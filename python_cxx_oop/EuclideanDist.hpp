#include "math.h"

class EuclideanDist
{
    private:
        double x0;
        double x1;
        double y0;
        double y1;

    public:
        EuclideanDist(double x0, double y0, double x1, double y1);
        double process();
        double operator()();
};

EuclideanDist::EuclideanDist(double x0, double y0, double x1, double y1)
{
    this->x0 = x0;
    this->y0 = y0;
    this->x1 = x1;
    this->y1 = y1;
}

double EuclideanDist::process()
{
    return sqrt( pow(x0-x1, 2) + pow(y0-y1, 2) );
}

double EuclideanDist::operator()()
{
    return this->process();
}