#!/usr/bin/env gnuplot
set terminal epslatex size 4.5 in, 2.5 in
set output "figures/threads.tex"

set datafile separator ','

set xlabel 'Threads'
set ylabel 'Time (s)'
unset key

plot 'data/cpu_custom.csv' using 1:2 with lines
