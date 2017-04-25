//
// Created by Yunming Zhang on 4/25/17.
//

#ifndef GRAPHIT_INTRINSICS_H_H
#define GRAPHIT_INTRINSICS_H_H

#include <vector>

template <typename T>
T vec_sum(std::vector<T> input_vector){
    T output_sum = 0;
    for (T elem : input_vector){
        output_sum += elem;
    }
    return output_sum;
}

#endif //GRAPHIT_INTRINSICS_H_H
