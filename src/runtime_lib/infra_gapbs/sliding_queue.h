// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef SLIDING_QUEUE_H_
#define SLIDING_QUEUE_H_

#include <algorithm>

#include "platform_atomics.h"


/*
GAP Benchmark Suite
Class:  SlidingQueue
Author: Scott Beamer

Double-buffered queue so appends aren't seen until SlideWindow() called
 - Use QueueBuffer when used in parallel to avoid false sharing by doing
   bulk appends from thread-local storage
*/


template <typename T>
class QueueBuffer;

template <typename T>
class SlidingQueue {


  friend class QueueBuffer<T>;



 public:
    T *shared;
    size_t shared_out_start;
    size_t shared_out_end;
    size_t shared_in;

    //initial queue max size
    size_t max_size;

  explicit SlidingQueue(size_t shared_size) {
    shared = new T[shared_size];
    reset();

    max_size = shared_size;

  }

  ~SlidingQueue() {
    delete[] shared;
  }

  void push_back(T to_add) {
    shared[shared_in++] = to_add;
  }

  bool empty() const {
    return shared_out_start == shared_out_end;
  }

  void reset() {
    shared_out_start = 0;
    shared_out_end = 0;
    shared_in = 0;
  }

  void slide_window() {
    shared_out_start = shared_out_end;
    shared_out_end = shared_in;
  }

  typedef T* iterator;

  iterator begin() const {
    return shared + shared_out_start;
  }

  iterator end() const {
    return shared + shared_out_end;
  }

  size_t size() const {
    return end() - begin();
  }
};


template <typename T>
class QueueBuffer {
  size_t in;
  T *local_queue;
  SlidingQueue<T> &sq;
  const size_t local_size;

 public:
  explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
      : sq(master), local_size(given_size) {
    in = 0;
    local_queue = new T[local_size];
  }

  ~QueueBuffer() {
    delete[] local_queue;
  }

  void push_back(T to_add) {
    if (in == local_size)
      flush();
    local_queue[in++] = to_add;
  }

//    void flush() {
//      T *shared_queue = sq.shared;
//      size_t copy_start = fetch_and_add(sq.shared_in, in);
//      std::copy(local_queue, local_queue+in, shared_queue+copy_start);
//      in = 0;
//    }



  void flush() {
    T *shared_queue = sq.shared;
    size_t copy_start = fetch_and_add(sq.shared_in, in);
    size_t copy_end = copy_start + in;

    if (copy_start/sq.max_size == copy_end/sq.max_size){
      //if we don't need to wrap around the queue,
      std::copy(local_queue, local_queue+in, shared_queue + (copy_start % sq.max_size));
    } else {
      //if we need to deal with wrap around the queue
      // compute the portion that goes
      size_t second_chunk_size = copy_end % sq.max_size;
      size_t first_chunk_size = in - second_chunk_size;
      //copy the first chunk to the back of the shared queue, all the way till the end
      std::copy(local_queue, local_queue+first_chunk_size,
                shared_queue +  (copy_start % sq.max_size));
      //copy the second chunk (wrap around chunk) to the beginning of the shared queue
      std::copy(local_queue+first_chunk_size, local_queue+in, shared_queue);
    }
    in = 0;
  }
};

#endif  // SLIDING_QUEUE_H_
