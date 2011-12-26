#!/bin/bash

img_left='images/left.png'
img_right='images/right.png'
algorithms=( cpu_opencv cpu_custom gpu_opencv gpu_custom )
threads_max=48
repeats=20

for algorithm in ${algorithms[@]}; do
    file="data/$algorithm.csv"
    /bin/echo -n > $file

    for threads in `seq 1 $threads_max`; do
        export OMP_NUM_THREADS=$threads
        sum=0

        for repeat in `seq 1 $repeats`; do
            sample=`./stereo $img_left $img_right $algorithm $repeats`
            sum=`echo "$sum + $sample" | bc`
        done

        /bin/echo -n "$threads," >> $file
        echo "scale=5;$sum / $repeats" | bc >> $file
    done
done
