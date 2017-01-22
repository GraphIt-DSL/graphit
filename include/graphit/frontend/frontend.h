//
// Created by Yunming Zhang on 1/15/17.
//

#ifndef GRAPHIT_FRONTEND_H_H
#define GRAPHIT_FRONTEND_H_H

#include <vector>
#include <string>
#include <fstream>


#endif //GRAPHIT_FRONTEND_H_H
namespace graphit {

    class Frontend {
    public:
        int parseStream(std::istream &programStream);

        /// Parses, typechecks and turns a given Simit-formated string into Simit IR.
        int parseString(const std::string &programString);

        /// Parses, typechecks and turns a given Simit-formated file into Simit IR.
        int parseFile(const std::string &filename);
    };

}