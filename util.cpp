#include <cmath>
#include <sys/time.h>
#include "util.hpp"

timer_t timer(void)
{
    timer_t timer;
    gettimeofday(&timer, NULL);
    return timer;
}

double duration(timer_t begin, timer_t end)
{
    double const secs  = begin.tv_sec - end.tv_sec;
    double const usecs = begin.tv_usec - end.tv_usec;
    return fabs(secs + 1e-6 * usecs);
}
