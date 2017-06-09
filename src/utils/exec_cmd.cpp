//
// Created by Yunming Zhang on 6/8/17.
//

#include <graphit/utils/exec_cmd.h>

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