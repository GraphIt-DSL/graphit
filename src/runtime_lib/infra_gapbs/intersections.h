// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details
#ifndef INTERSECTIONS_H_
#define INTERSECTIONS_H_

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <fstream>
#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "bitmap.h"
#include "timer.h"


using namespace std;


/*
Runtime library for various set intersection methods
*/


//Naive set intersection method.
static size_t intersectSortedNodeSetNaive(NodeID *A, NodeID *B, size_t totalA, size_t totalB, NodeID dest=(NodeID)INT32_MAX) {

    size_t begin_a = 0;
    size_t begin_b = 0;
    size_t count = 0;

    // intersect the tail using scalar intersection
    while (begin_a < totalA && begin_b < totalB) {
        if (*(B + begin_b) > dest) break;
        if (*(A + begin_a) < *(B + begin_b)) {
            begin_a++;
        } else if (*(A + begin_a) > *(B + begin_b)) {
            begin_b++;
        } else {
            count++;
            begin_a++;
            begin_b++;
        }
    }
    return count;

}

static size_t intersectSortedNodeSetBitset(Bitmap* A, NodeID *B, size_t totalB) {

    size_t total = 0;

    for (size_t j = 0; j < totalB; j++) {
        if (A->get_bit(*(B + j))) {
            total++;
        }
    }

    return total;
}


inline long int binarySearch(NodeID *A, long int start, size_t total, NodeID *B, size_t offset) {

    long int left = start == -1 ? 0 : start;
    long int right = total - 1;
    NodeID target = *(B + offset);
    while (left <= right) {

        long int medium = left + ((right - left) >> 1);
        NodeID current = *(A + medium);


        if (current == target) {
            return left;
        }

        if (current < target) {
            left = medium + 1;
        } else {
            right = medium - 1;
        }

    }

    return -1;


}

//set intersection by looking up smaller arrays in big arrays
static size_t intersectSortedNodeSetBinarySearch(NodeID *A, NodeID *B, size_t totalA, size_t totalB) {

    size_t count = 0;
    long int start = 0;
    long int prevStart = 0;

    if (totalA >= totalB) {

        for (size_t j = 0; j < totalB; j++) {
            start = binarySearch(A, start, totalA, B, j);
            if (start >= 0) {
                prevStart = start;
                count++;

            } else {
                start = prevStart;
            }

        }

    } else {

        for (size_t j = 0; j < totalA; j++) {
            start = binarySearch(B, start, totalB, A, j);
            if (start >= 0) {
                prevStart = start;
                count++;

            } else {
                start = prevStart;
            }

        }

    }

    return count;


}

//set intersection based on Hiroshi method -> reference: https://dl.acm.org/citation.cfm?id=2735518
static size_t intersectSortedNodeSetHiroshi(NodeID *A, NodeID *B, size_t totalA, size_t totalB, NodeID dest=(NodeID)INT32_MAX) {

    // if we cannot load 3 elements at a time, we just do naive intersection.
    if (totalA <= 2 || totalB <= 2){
        return intersectSortedNodeSetNaive(A, B, totalA, totalB, dest);
    }

    size_t begin_a = 0;
    size_t begin_b = 0;
    size_t count = 0;

    while (true) {
        NodeID Bdat0 = *(B + begin_b);
        NodeID Bdat1 = *(B + begin_b + 1);
        NodeID Bdat2 = *(B + begin_b + 2);

        // if dest is specified we don't want to go beyond this point to avoid overcounting
        if (Bdat2 > dest) break;

        NodeID Adat0 = *(A + begin_a);
        NodeID Adat1 = *(A + begin_a + 1);
        NodeID Adat2 = *(A + begin_a + 2);

        if (Adat0 == Bdat2) {
            count++;
            goto advanceB; // no more pair
        } else if (Adat2 == Bdat0) {
            count++;
            goto advanceA; // no more pair
        } else if (Adat0 == Bdat0) {
            count++;
        } else if (Adat0 == Bdat1) {
            count++;
        } else if (Adat1 == Bdat0) {
            count++;
        }
        if (Adat1 == Bdat1) {
            count++;
        } else if (Adat1 == Bdat2) {
            count++;
            goto advanceB;
        } else if (Adat2 == Bdat1) {
            count++;
            goto advanceA;
        }
        if (Adat2 == Bdat2) {
            count++;
            goto advanceAB;
        } else if (Adat2 > Bdat2) goto advanceB;
        else goto advanceA;
        advanceA:
        begin_a += 3;
        if (begin_a >= totalA - 2) { break; } else { continue; }
        advanceB:
        begin_b += 3;
        if (begin_b >= totalB - 2) { break; } else { continue; }
        advanceAB:
        begin_a += 3;
        begin_b += 3;
        if (begin_a >= totalA - 2 || begin_b >= totalB - 2) { break; }
    }

    // intersect the tail using scalar intersection
    while (begin_a < totalA && begin_b < totalB) {

        if (*(B + begin_b) > dest) break;

        if (*(A + begin_a) < *(B + begin_b)) {
            begin_a++;
        } else if (*(A + begin_a) > *(B + begin_b)) {
            begin_b++;
        } else {
            count++;
            begin_a++;
            begin_b++;
        }
    }
    return count;

}

//set intersection where it skips multiple element in batch
static size_t intersectSortedNodeSetMultipleSkip(NodeID *A, NodeID *B, size_t totalA, size_t totalB, NodeID dest=(NodeID)INT32_MAX) {

    NodeID it_a = 0;
    size_t count = 0;
    for (NodeID i = 0; i < totalB; i++) {
        NodeID w = *(B + i);
        if (w > dest) break;

        while (it_a < totalA && *(A + it_a) < w) {
            it_a += 3;
        }
        //if exceeds the boundary, set it at the boundary
        if (it_a >= totalA) {
            it_a = totalA - 1;
        }
        if (*(A + it_a) == w) {
            count++;
        } else {
            //rollback by 2 to make sure we are not skipping any intersections
            it_a -= 2;
            if (it_a >= 0) {
                if (*(A + it_a) == w) {
                    count++;
                }
            }
            it_a++;
            if (it_a >= 0) {
                if (*(A + it_a) == w) {
                    count++;
                }
            }
            it_a++;
        }
    }
    return count;
}

static size_t intersectSortedNodeSetCombined(NodeID *A, NodeID *B, size_t totalA, size_t totalB, size_t sizeThreshold,
                                      double ratioThreshold, NodeID dest=(NodeID)INT32_MAX) {

    size_t count = 0;

    if (totalA > sizeThreshold && totalB > sizeThreshold &&
        (totalA > ratioThreshold * totalB || totalB > ratioThreshold * totalA)) {
        count += intersectSortedNodeSetHiroshi(A, B, totalA, totalB, dest);

    }
    //TODO there is some tricky overcounting if we switch A, B.
    else {
        count += intersectSortedNodeSetMultipleSkip(A, B, totalA, totalB, dest);
    }

    return count;
}
#endif