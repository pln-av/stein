optsetup = ${LATEXDIR}/options-setup.sty
mathsetup = ${LATEXDIR}/math-setup.tex
listsetup = ${LATEXDIR}/listings-setup.tex
inputs = garch1.md 
cpplibs.pdf : $(inputs)
	pandoc -s -o garch.pdf $(inputs) --listings -H $(optsetup) -H $(mathsetup) -H $(listsetup) --pdf-engine /Library/TeX/texbin/pdflatex

clean :
	rm -f garch.pdf
