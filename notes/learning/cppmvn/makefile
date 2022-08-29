optsetup = ${LATEXDIR}/options-setup.sty
mathsetup = ${LATEXDIR}/math-setup.tex
listsetup = ${LATEXDIR}/listings-setup.tex
inputs = cppmvn.md 
cppmvn.pdf : $(inputs)
	pandoc -s -o cppmvn.pdf $(inputs) --listings -H $(optsetup) -H $(mathsetup) -H $(listsetup) --pdf-engine /Library/TeX/texbin/pdflatex

clean :
	rm -f cppmvn.pdf
