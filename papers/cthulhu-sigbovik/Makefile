.PHONY: all clean view

MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.pdf

view: $(MAIN).pdf
	open $(MAIN).pdf
