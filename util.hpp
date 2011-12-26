#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <sys/time.h>

typedef struct timeval timeval_t;

timeval_t timer(void);
double duration(timeval_t start, timeval_t end);

#endif
