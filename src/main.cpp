#include <graphit/frontend/scanner.h>
#include <fstream>


    int main() {

        graphit::Scanner *sc = new graphit::Scanner();
        sc->lex();
        return 0;

    }

