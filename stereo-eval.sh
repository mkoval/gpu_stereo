#!/bin/sh

img_left='images/left.png'
img_right='images/right.png'
threads_max=48
repeats=20


for threads in $(seq 1 $threads_max); do
    export OMP_NUM_THREADS=$threads
    echo -n $threads

    for repeat in $(seq 1 $repeats); do
        echo -n ,
        ./stereo $img_left $img_right cpu_custom $repeats
    done
done
