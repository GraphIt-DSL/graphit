//
// Created by Yunming Zhang on 5/31/17.
//

// Code from Stackoverflow
// https://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix

#ifndef GRAPHIT_EXECUTE_CMD_H_H
#define GRAPHIT_EXECUTE_CMD_H_H

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

std::string exec_cmd(std::string input_cmd) {
    const char* cmd = input_cmd.c_str();
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

#endif //GRAPHIT_EXECUTE_CMD_H_H
