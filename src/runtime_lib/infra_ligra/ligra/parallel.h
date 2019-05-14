// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _PARALLEL_H
#define _PARALLEL_H

#if defined(CILK)
#include <cilk/cilk.h>
#define parallel_main main
namespace ligra {
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, IterT step, BodyT body) {
  cilk_for(IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("cilk_grainsize = 1") cilk_for(IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("cilk_grainsize = 256") cilk_for(IterT i = start; i < end; i += step)
    body(i);
}
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, BodyT body) {
  parallel_for(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, BodyT body) {
  parallel_for_1(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, BodyT body) {
  parallel_for_256(start, end, (IterT)1, body);
}
}
#include <cilk/cilk_api.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
static int getWorkers() {
  return __cilkrts_get_nworkers();
}
static void setWorkers(int n) {
  __cilkrts_end_cilk();
  //__cilkrts_init();
  std::stringstream ss; ss << n;
  if (0 != __cilkrts_set_param("nworkers", ss.str().c_str())) {
    std::cerr << "failed to set worker count!" << std::endl;
    std::abort();
  }
}

// intel cilk+
#elif defined(CILKP)
#include <cilk/cilk.h>
namespace ligra {
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, IterT step, BodyT body) {
  cilk_for(IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("cilk_grainsize = 1") cilk_for(IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("cilk_grainsize = 256") cilk_for(IterT i = start; i < end; i += step)
    body(i);
}
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, BodyT body) {
  parallel_for(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, BodyT body) {
  parallel_for_1(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, BodyT body) {
  parallel_for_256(start, end, (IterT)1, body);
}
}
#define parallel_main main
#include <cilk/cilk_api.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
static int getWorkers() {
  return __cilkrts_get_nworkers();
}
static void setWorkers(int n) {
  __cilkrts_end_cilk();
  //__cilkrts_init();
  std::stringstream ss; ss << n;
  if (0 != __cilkrts_set_param("nworkers", ss.str().c_str())) {
    std::cerr << "failed to set worker count!" << std::endl;
    std::abort();
  }
}

// intel TBB
#elif defined(TBB)
#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"
namespace ligra {
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, IterT step, BodyT body) {
  tbb::parallel_for(start, end, step, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, IterT step, BodyT body) {
  // ignore grain size for now
  tbb::parallel_for(start, end, step, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, IterT step, BodyT body) {
  // ignore grain size for now
  tbb::parallel_for(start, end, step, body);
}
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, BodyT body) {
  parallel_for(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, BodyT body) {
  parallel_for_1(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, BodyT body) {
  parallel_for_256(start, end, (IterT)1, body);
}
}
#define parallel_main main
static int s_nthreads;
static int getWorkers() {
  return s_nthreads;
}
static void setWorkers(int n) {
  tbb::task_scheduler_init init(n);
  s_nthreads = n;
}

// openmp
#elif defined(OPENMP)
#include <omp.h>
#define cilk_spawn
#define cilk_sync
#define parallel_main main
namespace ligra {
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("omp parallel for ") for (IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("omp parallel for schedule (static,1)") for (IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, IterT step, BodyT body) {
  _Pragma("omp parallel for schedule (static,256)") for (IterT i = start; i < end; i += step)
    body(i);
}
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, BodyT body) {
  parallel_for(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, BodyT body) {
  parallel_for_1(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, BodyT body) {
  parallel_for_256(start, end, (IterT)1, body);
}
}
//#define parallel_for _Pragma("omp parallel for ") for
//#define parallel_for _Pragma("omp parallel for schedule (dynamic, 64)") for
static int getWorkers() { return omp_get_max_threads(); }
static void setWorkers(int n) { omp_set_num_threads(n); }

// c++
#else
#define cilk_spawn
#define cilk_sync
#define parallel_main main
namespace ligra {
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, IterT step, BodyT body) {
  for (IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, IterT step, BodyT body) {
  for (IterT i = start; i < end; i += step)
    body(i);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, IterT step, BodyT body) {
  for (IterT i = start; i < end; ++i)
    body(i);
}
template<typename IterT, typename BodyT>
void parallel_for(IterT start, IterT end, BodyT body) {
  parallel_for(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_1(IterT start, IterT end, BodyT body) {
  parallel_for_1(start, end, (IterT)1, body);
}

template<typename IterT, typename BodyT>
void parallel_for_256(IterT start, IterT end, BodyT body) {
  parallel_for_256(start, end, (IterT)1, body);
}
}
#define cilk_for for
static int getWorkers() { return 1; }
static void setWorkers(int n) { }

#endif

#include <limits.h>

#define LONG
#if defined(LONG)
typedef long intT;
typedef unsigned long uintT;
#define INT_T_MAX LONG_MAX
#define UINT_T_MAX ULONG_MAX
#else
typedef int intT;
typedef unsigned int uintT;
#define INT_T_MAX INT_MAX
#define UINT_T_MAX UINT_MAX
#endif

//edges store 32-bit quantities unless EDGELONG is defined
#if defined(EDGELONG)
typedef long intE;
typedef unsigned long uintE;
#define INT_E_MAX LONG_MAX
#define UINT_E_MAX ULONG_MAX
#else
typedef int intE;
typedef unsigned int uintE;
#define INT_E_MAX INT_MAX
#define UINT_E_MAX UINT_MAX
#endif

#endif // _PARALLEL_H
