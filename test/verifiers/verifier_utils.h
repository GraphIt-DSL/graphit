//
// Created by Yunming Zhang on 7/13/17.
//

#ifndef GRAPHIT_VERIFIER_UTILS_H
#define GRAPHIT_VERIFIER_UTILS_H

template <typename T>
pvector<T>* readFileIntoVector(std::string file_name){

    std::ifstream file(file_name);
    pvector<T>* output = new pvector<T>();
    T u;
    while (file >> u) {
        output->push_back(u);
    }
    return output;
}

template <typename T>
T readFileIntoSize(std::string file_name) {
    std::ifstream file(file_name);
    T output;
    file >> output;
    return output;

}

#endif //GRAPHIT_VERIFIER_UTILS_H
