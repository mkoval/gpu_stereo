#include <cmath>
#include <sys/time.h>
#include "util.hpp"

timeval_t timer(void)
{
    timeval_t timer;
    gettimeofday(&timer, 0);
    return timer;
}

double duration(timeval_t begin, timeval_t end)
{
    double const secs  = begin.tv_sec - end.tv_sec;
    double const usecs = begin.tv_usec - end.tv_usec;
    return fabs(secs + 1e-6 * usecs);
}
