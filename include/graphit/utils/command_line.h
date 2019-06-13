//
// Created by Yunming Zhang on 4/10/17.
//

#ifndef GRAPHIT_COMMAND_LINE_H
#define GRAPHIT_COMMAND_LINE_H

#include <getopt.h>

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <string>
#include <vector>

/**
 * Code borrowed from GAPBS benchmark project by Scott Beamer
 */

class CLBase {
protected:
    int argc_;
    char** argv_;
    std::string name_;
    // f: means -f flag requires a follow on name,
    std::string get_args_ = "f:o:p:m:h";
    std::vector<std::string> help_strings_;
    std::string input_filename_ = "";
    std::string output_filename_ = "";
    std::string python_module_path_ = "";
    std::string python_module_name_ = ""; 


    void AddHelpLine(char opt, std::string opt_arg, std::string text,
                     std::string def = "") {
        const int kBufLen = 100;
        char buf[kBufLen];
        if (opt_arg != "")
            opt_arg = "<" + opt_arg + ">";
        if (def != "")
            def = "[" + def + "]";
        snprintf(buf, kBufLen, " -%c %-9s: %-54s%10s", opt, opt_arg.c_str(),
                 text.c_str(), def.c_str());
        help_strings_.push_back(buf);
    }

public:
    CLBase(int argc, char** argv, std::string name = "") :
            argc_(argc), argv_(argv), name_(name) {
        AddHelpLine('h', "", "print this help message");
        AddHelpLine('f', "file", "input file");
        AddHelpLine('o', "", "output file");
	AddHelpLine('p', "", "Python module path");
	AddHelpLine('m', "", "Python module name");
    }

    bool ParseArgs() {
        signed char c_opt;
        extern char *optarg;          // from and for getopt
        while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
            HandleArg(c_opt, optarg);
        }
        if (input_filename_ == "") {
            std::cout << "No input file specified. (Use -h for help)" << std::endl;
            return false;
        }
        if (output_filename_ == "") {
            std::cout << "No output file specified. (Use -h for help)" << std::endl;
            return false;
        }

        return true;
    }

    void virtual HandleArg(signed char opt, char* opt_arg) {
        switch (opt) {
            case 'f': input_filename_ = std::string(opt_arg);           break;
            case 'o': output_filename_ = std::string(opt_arg);                     break;
	    case 'm': python_module_name_ = std::string(opt_arg); break;
	    case 'p': python_module_path_ = std::string(opt_arg); break;
            case 'h': PrintUsage();                               break;
        }
    }

    void PrintUsage() {
        std::cout << name_ << std::endl;
        // std::sort(help_strings_.begin(), help_strings_.end());
        for (std::string h : help_strings_)
            std::cout << h << std::endl;
        std::exit(0);
    }

    std::string input_filename() const { return input_filename_; }
    std::string output_filename() const { return output_filename_; }
    std::string python_module_path() const { return python_module_path_; }
    std::string python_module_name() const { return python_module_name_; }
};


#endif //GRAPHIT_COMMAND_LINE_H
