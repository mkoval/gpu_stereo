#!/usr/bin/env gnuplot
set terminal epslatex size 2.75 in, 1.75 in
set output "figures/comparison.tex"

set boxwidth 1 relative
set style data histograms
set style fill solid 1.0 border -1
set datafile separator ','

set ylabel 'Time (s)'

plot './data/comparison.csv' using 2:xtic(1) with boxes
