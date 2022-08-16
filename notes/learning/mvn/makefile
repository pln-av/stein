optsetup = ${LATEXDIR}/options-setup.sty
mathsetup = ${LATEXDIR}/math-setup.tex
listsetup = ${LATEXDIR}/listings-setup.tex
inputs = mvn.md mvn-examples.md
mvn.pdf : mvn.md 
	pandoc -s -o mvn.pdf $(inputs) --listings -H $(optsetup) -H $(mathsetup) -H $(listsetup) --pdf-engine /Library/TeX/texbin/pdflatex

clean :
	rm mvn.pdf
