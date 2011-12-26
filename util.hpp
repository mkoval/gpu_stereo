#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <sys/time.h>

typedef struct timeval timer_t;

timer_t timer(void);
double duration(timer_t start, timer_t end);

#endif
