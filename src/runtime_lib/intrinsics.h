//
// Created by Yunming Zhang on 4/25/17.
//

#ifndef GRAPHIT_INTRINSICS_H_H
#define GRAPHIT_INTRINSICS_H_H

#include <vector>

#include "infra_gapbs/builder.h"
#include "infra_gapbs/benchmark.h"
#include "infra_gapbs/bitmap.h"
#include "infra_gapbs/command_line.h"
#include "infra_gapbs/graph.h"
#include "infra_gapbs/platform_atomics.h"
#include "infra_gapbs/pvector.h"
#include <queue>
#include <curses.h>
#include "infra_gapbs/timer.h"
#include "infra_gapbs/sliding_queue.h"

#include "vertexsubset.h"

#include <time.h>
#include <chrono>

template <typename T>
T builtin_sum(std::vector<T> input_vector){
    T output_sum = 0;
    for (T elem : input_vector){
        output_sum += elem;
    }
    return output_sum;
}

Graph builtin_loadEdgesFromFile(std::string file_name){
    CLBase cli (file_name);
    Builder builder (cli);
    Graph g = builder.MakeGraph();
    return g;
}

int builtin_getVertices(Graph &edges){
    return edges.num_nodes();
}

std::vector<int> builtin_getOutDegrees(Graph &edges){
    std::vector<int> out_degrees (edges.num_nodes(), 0);
    for (NodeID n=0; n < edges.num_nodes(); n++){
        out_degrees[n] = edges.out_degree(n);
    }
    return out_degrees;
}

//float getTime(){
//    using namespace std::chrono;
//    auto t = high_resolution_clock::now();
//    time_point<high_resolution_clock,microseconds> usec = time_point_cast<microseconds>(t);
//    return (float)(usec.time_since_epoch().count())/1000;
//}

struct timeval start_time_;
struct timeval elapsed_time_;

void startTimer(){
    gettimeofday(&start_time_, NULL);
}

float stopTimer(){
    gettimeofday(&elapsed_time_, NULL);
    elapsed_time_.tv_sec  -= start_time_.tv_sec;
    elapsed_time_.tv_usec -= start_time_.tv_usec;
    return elapsed_time_.tv_sec + elapsed_time_.tv_usec/1e6;

}



#endif //GRAPHIT_INTRINSICS_H_H
