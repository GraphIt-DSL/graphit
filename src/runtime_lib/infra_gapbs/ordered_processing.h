#ifndef ORDERED_PROCESSING_H_
#define ORDERED_PROCESSING_H_

#include "graph.h"
#include "eager_priority_queue.h"


using namespace std;

const size_t kMaxBin = std::numeric_limits<size_t>::max()/2;



template <typename PriorityT_>
struct updatePriorityMin
{
  void operator()(EagerPriorityQueue<PriorityT_>* pq, 
  					vector<vector<NodeID> >& local_bins,
  					NodeID dst, PriorityT_ old_val, 
		  PriorityT_ new_val){
    if (new_val < old_val) {
      bool changed_dist = true;
      while (!compare_and_swap(pq->priorities_[dst], old_val, new_val)) {
        old_val = pq->priorities_[dst];
        if (old_val <= new_val) {
          changed_dist = false;
          break;
        }
      }
      if (changed_dist) {
      	// assume the priority is mapped to a bin using delta
      	size_t dest_bin;
      	if (pq->delta_ != 1) dest_bin = new_val/pq->delta_;
      	else dest_bin = new_val;
      	
        if (dest_bin >= local_bins.size()) {
	  local_bins.resize(dest_bin+1);
        }
        local_bins[dest_bin].push_back(dst);
      }
    }
  }
};


template< class Priority, class EdgeApplyFunc , class WhileCond>
  void OrderedProcessingOperatorNoMerge(EagerPriorityQueue<Priority>* pq, const WGraph &g, WhileCond while_cond, EdgeApplyFunc edge_apply,  NodeID optional_source_node){

  pvector<NodeID> frontier(g.num_edges_directed());
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  //size_t shared_indexes[2] = {0, kMaxBin};
  //size_t frontier_tails[2] = {1, 0};

  pq->init_indexes_tails();
  
  //optional source node


  frontier[0] = optional_source_node;

//  int round = 0;
  
  #pragma omp parallel
  {
    vector<vector<NodeID> > local_bins(0);
    size_t iter = 0;
    while (while_cond()) {
      //TODO: refactor to use user supplied 
      // while (user_supplied_condition())

      // size_t &curr_bin_index = shared_indexes[iter&1];
//       size_t &next_bin_index = shared_indexes[(iter+1)&1];
//       size_t &curr_frontier_tail = frontier_tails[iter&1];
//       size_t &next_frontier_tail = frontier_tails[(iter+1)&1];



      size_t &curr_bin_index = pq->shared_indexes[iter&1];
      size_t &next_bin_index = pq->shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = pq->frontier_tails[iter&1];
      size_t &next_frontier_tail = pq->frontier_tails[(iter+1)&1];

// Used for debugging
//      round++;
//      std::cout << " round: " << round << std::endl;
//      std::cout << " frontier size: " << curr_frontier_tail << std::endl;
//      for (size_t i=0; i < curr_frontier_tail; i++) {
//          NodeID u = frontier[i];
//          std::cout << u << " ";
//      }
//      std::cout << std::endl;

      #pragma omp for nowait schedule(dynamic, 64)
      for (size_t i=0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
	//TODO: need to refactor to use user supplied filtering on the source node
        //if (src_filter(u)) { //hard code this into the library
	if (pq->priorities_[u] >= pq->delta_*pq->get_current_priority()){
          for (WNode wn : g.out_neigh(u)) {
             edge_apply(local_bins, u, wn.v, wn.w);
          }
 
    } //end of if statement
    }//going through current frontier for end

      //searching for the next priority

      for (size_t i=pq->get_current_priority(); i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
      //t.Stop();
      //PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
      //  t.Start();
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
	// need to make srue we increment it from only one thread
	pq->increment_iter();
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      
      #pragma omp barrier
    }
    //#pragma omp single
    //cout << "order processing took " << iter << " iterations" << endl;
  }//end of pragma omp parallel   
  

}



template<class Priority,  class WhileCond, class EdgeApplyFunc >
  void OrderedProcessingOperatorWithMerge(EagerPriorityQueue<Priority>* pq, const WGraph &g,  WhileCond while_cond, EdgeApplyFunc edge_apply, int bin_size_threshold = 1000, NodeID optional_source_node=-1){

  pvector<NodeID> frontier(g.num_edges_directed());
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  //size_t shared_indexes[2] = {0, kMaxBin};
  //size_t frontier_tails[2] = {1, 0};

  pq->init_indexes_tails();
  
  //optional source node


  frontier[0] = optional_source_node;
  
  #pragma omp parallel
  {
    vector<vector<NodeID> > local_bins(0);
    size_t iter = 0;
    while (while_cond()) {
      //TODO: refactor to use user supplied 
      // while (user_supplied_condition())

      // size_t &curr_bin_index = shared_indexes[iter&1];
//       size_t &next_bin_index = shared_indexes[(iter+1)&1];
//       size_t &curr_frontier_tail = frontier_tails[iter&1];
//       size_t &next_frontier_tail = frontier_tails[(iter+1)&1];


      size_t &curr_bin_index = pq->shared_indexes[iter&1];
      size_t &next_bin_index = pq->shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = pq->frontier_tails[iter&1];
      size_t &next_frontier_tail = pq->frontier_tails[(iter+1)&1];

      #pragma omp for nowait schedule(dynamic, 64)
      for (size_t i=0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
	//TODO: need to refactor to use user supplied filtering on the source node
        //if (src_filter(u)) {
	if (pq->priorities_[u] >= pq->delta_*pq->get_current_priority()){
          for (WNode wn : g.out_neigh(u)) {
             edge_apply(local_bins, u, wn.v, wn.w);
          }
 
    } //end of if statement
    }//going through current frontier for end



      while (local_bins.size() > 0 && curr_bin_index < local_bins.size() && !local_bins[curr_bin_index].empty()){
      size_t cur_bin_size = local_bins[curr_bin_index].size();
      if (cur_bin_size > bin_size_threshold) break;

        vector<NodeID> cur_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        for (size_t i=0; i < cur_bin_size; i++) {
          NodeID u = cur_bin_copy[i];
          //if (src_filter(u)) {
	  if (pq->priorities_[u] >= pq->delta_*pq->get_current_priority()){
              for (WNode wn : g.out_neigh(u)) {  
                 edge_apply(local_bins, u, wn.v, wn.w);
              }
          }
        }
    }
      //searching for the next priority

      for (size_t i=pq->get_current_priority(); i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
      //t.Stop();
      //PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
      //  t.Start();
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
	// need to make srue we increment it from only one thread
	pq->increment_iter();
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      
      #pragma omp barrier
    }
    //#pragma omp single
    //cout << "took " << iter << " iterations" << endl;
  }//end of pragma omp parallel   
  

}

#endif  // ORDERED_PROCESSING_H
