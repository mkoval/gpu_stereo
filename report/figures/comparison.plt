#!/usr/bin/env gnuplot
set terminal epslatex size 4.5 in, 2.5 in
set output "figures/comparison.tex"

set boxwidth 1 relative
set style data histogram
set boxwidth 0.75 absolute
set style fill solid 1.0 border -1
set format x "\\footnotesize %g"

set datafile separator ','

unset key

set ylabel 'Average Time (s)'

plot './data/comparison.csv' using 2:xticlabels("{\\footnotesize " . stringcolumn(1) . "}") with boxes
