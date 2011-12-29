#!/usr/bin/env gnuplot
set terminal epslatex size 2.75 in, 1.75 in
set output "figures/threads.tex"

set datafile separator ','

set xlabel 'Threads'
set ylabel 'Time (s)'
unset key

plot 'data/cpu_custom.csv' using 1:2 with lines
