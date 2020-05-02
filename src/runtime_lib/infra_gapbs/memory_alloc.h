//
// Created by Tugsbayasgalan Manlaibaatar on 2020-05-02.
//

#ifndef GRAPHIT_MEMORY_ALLOC_H
#define GRAPHIT_MEMORY_ALLOC_H

#include <stdio.h>
#include <cinttypes>
#include <iostream>
#include <type_traits>
#include <map>
#include <thread>
#include <mutex>

#include "pvector.h"
#include "util.h"

#include "infra_ligra/ligra/parallel.h"

#include "segmentgraph.h"
#include <memory>
#include <assert.h>



class MemAlloc {

public:

    MemAlloc(int64_t numNodes) : num_nodes_(numNodes){}

    ~MemAlloc() {
        for (int i = 0; i < deduplication_flags.size();i++) {
            delete[] deduplication_flags[i];
        }
        deduplication_flags.clear();
    }

    inline int* get_flags_atomic_() {
        static std::mutex thread_mutex;
        std::lock_guard<std::mutex> lock(thread_mutex);

        if (deduplication_flags.size() == 0) {

            deduplication_flags.push_back(new int[num_nodes_]);
            ligra::parallel_for_lambda(0, (int)num_nodes_, [&] (int i) { deduplication_flags.back()[i]=0; });
        }

        int * to_return = deduplication_flags.back();

        deduplication_flags.pop_back();

        return to_return;

    }

    inline void return_flags_atomic_(int * flags) {

        static std::mutex thread_mutex;
        std::lock_guard<std::mutex> lock(thread_mutex);
        deduplication_flags.push_back(flags);

    }



private:
    std::vector<int*> deduplication_flags;
    int64_t num_nodes_;
};

#endif //GRAPHIT_MEMORY_ALLOC_H
