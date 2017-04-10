#include <graphit/frontend/scanner.h>
#include <graphit/midend/midend.h>
#include <fstream>
#include <graphit/frontend/frontend.h>

#include <graphit/utils/command_line.h>
#include <graphit/backend/backend.h>
#include <graphit/frontend/error.h>
#include <fstream>

using namespace graphit;

int main(int argc, char* argv[]) {
    // Set up various data structures
    CLBase cli(argc, argv, "graphit compiler");
    graphit::FIRContext* context = new graphit::FIRContext();
    std::vector<ParseError> * errors = new std::vector<ParseError>();
    Frontend * fe = new Frontend();
    graphit::MIRContext* mir_context  = new graphit::MIRContext();

    //parse the arguments
    if (!cli.ParseArgs())
        return -1;

    //read input file into buffer
    std::ifstream file(cli.input_filename());
    std::stringstream buffer;
    if(!file) {
        std::cout << "error reading the input file" << std::endl;
    }
    buffer << file.rdbuf();
    file.close();

    //set up the output file
    std::ofstream output_file;
    output_file.open(cli.output_filename());

    //compile the input file
    fe->parseStream(buffer, context, errors);
    graphit::Midend* me = new graphit::Midend(context);
    me->emitMIR(mir_context);
    graphit::Backend* be = new graphit::Backend(mir_context);
    be->emitCPP(output_file);

    output_file.close();

    return 0;

}

