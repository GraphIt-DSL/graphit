#include <graphit/frontend/scanner.h>
#include <graphit/midend/midend.h>
#include <fstream>
#include <graphit/frontend/frontend.h>

#include <graphit/utils/command_line.h>

int main(int argc, char* argv[]) {
    CLBase cli(argc, argv, "graphit compiler");
    if (!cli.ParseArgs())
        return -1;
    graphit::Frontend* frontend = new graphit::Frontend();
    //graphit::Midend* midend = new graphit::Midend();
    return 0;

}

