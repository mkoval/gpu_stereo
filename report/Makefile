BIBTEX   = bibtex
GNUPLOT  = gnuplot
PDFLATEX = pdflatex
TARGET   = report
FIGURES  = figures/comparison.tex \
           figures/threads.tex
TRASH    = $(TARGET).aux \
           $(TARGET).log \
           $(TARGET).pdf \
           $(TARGET).bbl \
           $(TARGET).blg

.PHONY: all clean
.SUFFIXES:
.SECONDARY:

$(TARGET).pdf: $(FIGURES)

all: $(TARGET).pdf

clean:
	$(RM) $(TRASH) $(FIGURES:.plt=.tex) $(FIGURES:.plt=.eps)

%.pdf: %.tex %.bib
	$(PDFLATEX) -interaction=nonstopmode $(basename $<)
	$(BIBTEX) $(basename $<)
	$(PDFLATEX) -interaction=nonstopmode $(basename $<)
	$(PDFLATEX) -interaction=nonstopmode $(basename $<)

figures/%.tex: figures/%.plt data/%.csv
	$(GNUPLOT) $<
