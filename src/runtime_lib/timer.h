#ifndef GRAPHIT_TIMER_H
#define GRAPHIT_TIMER_H
#include <sys/time.h>

static struct timeval start_time_;
static struct timeval elapsed_time_;

static void startTimer(){
    gettimeofday(&start_time_, NULL);
}

static float stopTimer(){
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
    return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;

}
#endif
