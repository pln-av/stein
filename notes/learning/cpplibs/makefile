optsetup = ${LATEXDIR}/options-setup.sty
mathsetup = ${LATEXDIR}/math-setup.tex
listsetup = ${LATEXDIR}/listings-setup.tex
inputs = cpplibs.md 
cpplibs.pdf : $(inputs)
	pandoc -s -o cpplibs.pdf $(inputs) --listings -H $(optsetup) -H $(mathsetup) -H $(listsetup) --pdf-engine /Library/TeX/texbin/pdflatex

clean :
	rm -f cpplibs.pdf
