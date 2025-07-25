# Usa pdflatex
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error';

# Activa bibtex automáticamente
$bibtex_use = 1;

# Activa makeindex para índices
$makeindex = 'makeindex %O -o %D %S';

# Activa makeglossaries para glosarios si los usas
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
sub makeglossaries {
    return system("makeglossaries $_[0]");
}
