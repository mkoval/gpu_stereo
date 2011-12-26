#!/bin/sh

img_left='images/left.png'
img_right='images/right.png'
threads_max=2
repeats=3

for threads in `seq 1 $threads_max`; do
    export OMP_NUM_THREADS=$threads
    sum=0

    for repeat in `seq 1 $repeats`; do
        sample=`./stereo $img_left $img_right cpu_custom $repeats`
        sum=`echo "$sum + $sample" | bc`
    done

    /bin/echo -n "$threads,"
    echo "scale=5;$sum / $repeats" | bc
done
