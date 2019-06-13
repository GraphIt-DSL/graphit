// copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdio.h>
#include <cinttypes>
#include <iostream>
#include <type_traits>
#include <map>

#include "pvector.h"
#include "util.h"

#include "segmentgraph.h"
#include <memory>
#include <assert.h>

/*
GAP Benchmark Suite
Class:  CSRGraph
Author: Scott Beamer

Simple container for graph in CSR format
 - Intended to be constructed by a Builder
 - To make weighted, set DestID_ template type to NodeWeight
 - MakeInverse parameter controls whether graph stores its inverse
*/


// Used to hold node & weight, with another node it makes a weighted edge
template <typename NodeID_=int32_t, typename WeightT_=int32_t>
struct NodeWeight {
  NodeID_ v;
  WeightT_ w;
  NodeWeight() {}
  NodeWeight(NodeID_ v) : v(v), w(1) {}
  NodeWeight(NodeID_ v, WeightT_ w) : v(v), w(w) {}

  bool operator< (const NodeWeight& rhs) const {
    return v == rhs.v ? w < rhs.w : v < rhs.v;
  }

  // doesn't check WeightT_s, needed to remove duplicate edges
  bool operator== (const NodeWeight& rhs) const {
    return v == rhs.v;
  }

  // doesn't check WeightT_s, needed to remove self edges
  bool operator== (const NodeID_& rhs) const {
    return v == rhs;
  }

  operator NodeID_() {
    return v;
  }
};

template <typename NodeID_, typename WeightT_>
std::ostream& operator<<(std::ostream& os,
                         const NodeWeight<NodeID_, WeightT_>& nw) {
  os << nw.v << " " << nw.w;
  return os;
}

template <typename NodeID_, typename WeightT_>
std::istream& operator>>(std::istream& is, NodeWeight<NodeID_, WeightT_>& nw) {
  is >> nw.v >> nw.w;
  return is;
}



// Syntatic sugar for an edge
template <typename SrcT, typename DstT = SrcT>
struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int32_t SGID;
typedef EdgePair<SGID> SGEdge;
typedef int64_t SGOffset;



template <class NodeID_, class DestID_ = NodeID_, bool MakeInverse = true>
class CSRGraph {
  // Used to access neighbors of vertex, basically sugar for iterators
  class Neighborhood {
    NodeID_ n_;
    DestID_** g_index_;
   public:
    Neighborhood(NodeID_ n, DestID_** g_index) : n_(n), g_index_(g_index) {}
    typedef DestID_* iterator;
    iterator begin() { return g_index_[n_]; }
    iterator end()   { return g_index_[n_+1]; }
  };

  void ReleaseResources() {
    //added a second condition to prevent double free (transpose graphs)
/*
    if (out_index_ != nullptr)
      delete[] out_index_;
    if (out_neighbors_ != nullptr)
      delete[] out_neighbors_;
    if (directed_) {
      if (in_index_ != nullptr && in_index_ != out_index_)
        delete[] in_index_;
      if (in_neighbors_ != nullptr && in_neighbors_ != out_neighbors_)
        delete[] in_neighbors_;
    }
    if (flags_ != nullptr)
      delete[] flags_;
*/
    out_index_shared_.reset();
    out_neighbors_shared_.reset();
    in_index_shared_.reset();
    in_neighbors_shared_.reset();
    flags_shared_.reset();
    offsets_shared_.reset();
    for (auto iter = label_to_segment.begin(); iter != label_to_segment.end(); iter++) {
      delete ((*iter).second);
    }
  }


 public:
  julienne::graph<julienne::symmetricVertex> julienne_graph = __julienne_null_graph;
  //julienne::EdgeMap<julienne::uintE, julienne::symmetricVertex> *em;
  CSRGraph() : directed_(false), num_nodes_(-1), num_edges_(-1),
    out_index_(nullptr), out_neighbors_(nullptr),
  in_index_(nullptr), in_neighbors_(nullptr), flags_(nullptr), is_transpose_(false) {}

  CSRGraph(int64_t num_nodes, DestID_** index, DestID_* neighs) :
    directed_(false), num_nodes_(num_nodes),
    out_index_(index), out_neighbors_(neighs),
    in_index_(index), in_neighbors_(neighs){
      out_index_shared_.reset(index);
      out_neighbors_shared_.reset(neighs);
      in_index_shared_ = out_index_shared_;
      in_neighbors_shared_ = out_neighbors_shared_;
      
      num_edges_ = (out_index_[num_nodes_] - out_index_[0]) / 2;
      //adding flags used for deduplication
      flags_ = new int[num_nodes_];
      flags_shared_.reset(flags_);
    //adding offsets for load balacne scheme
    SetUpOffsets(true);
    //Set this up for getting random neighbors
      srand(time(NULL));
    }

  CSRGraph(int64_t num_nodes, DestID_** out_index, DestID_* out_neighs,
        DestID_** in_index, DestID_* in_neighs) :
    directed_(true), num_nodes_(num_nodes),
    out_index_(out_index), out_neighbors_(out_neighs),
    in_index_(in_index), in_neighbors_(in_neighs), is_transpose_(false){
      num_edges_ = out_index_[num_nodes_] - out_index_[0];
      out_index_shared_.reset(out_index);
      out_neighbors_shared_.reset(out_neighs);
      in_index_shared_.reset(in_index);
      in_neighbors_shared_.reset(in_neighs);
      flags_ = new int[num_nodes_];
      flags_shared_.reset(flags_);
        SetUpOffsets(true);

      //Set this up for getting random neighbors
      srand(time(NULL));
    }

    CSRGraph(int64_t num_nodes, DestID_** out_index, DestID_* out_neighs,
        DestID_** in_index, DestID_* in_neighs, bool is_transpose) :
    directed_(true), num_nodes_(num_nodes),
    out_index_(out_index), out_neighbors_(out_neighs),
    in_index_(in_index), in_neighbors_(in_neighs) , is_transpose_(is_transpose){
      num_edges_ = out_index_[num_nodes_] - out_index_[0];

      out_index_shared_.reset(out_index);
      out_neighbors_shared_.reset(out_neighs);
      in_index_shared_.reset(in_index);
      in_neighbors_shared_.reset(in_neighs);
      flags_ = new int[num_nodes_];
      flags_shared_.reset(flags_);
      SetUpOffsets(true);

        //Set this up for getting random neighbors
        srand(time(NULL));
    }

    CSRGraph(int64_t num_nodes, std::shared_ptr<DestID_*> out_index, std::shared_ptr<DestID_> out_neighs,
        shared_ptr<DestID_*> in_index, shared_ptr<DestID_> in_neighs, bool is_transpose) :
    directed_(true), num_nodes_(num_nodes),
    out_index_(out_index.get()), out_neighbors_(out_neighs.get()),
    in_index_(in_index.get()), in_neighbors_(in_neighs.get()) , is_transpose_(is_transpose){
      num_edges_ = out_index_[num_nodes_] - out_index_[0];

      out_index_shared_ = (out_index);
      out_neighbors_shared_ = (out_neighs);
      in_index_shared_ = (in_index);
      in_neighbors_shared_ = (in_neighs);
      flags_ = new int[num_nodes_];
      flags_shared_.reset(flags_);
    SetUpOffsets(true);
        //Set this up for getting random neighbors
        srand(time(NULL));
  }

  
    CSRGraph(CSRGraph& other) : directed_(other.directed_),
                                 num_nodes_(other.num_nodes_), num_edges_(other.num_edges_),
                                 out_index_(other.out_index_), out_neighbors_(other.out_neighbors_),
                                 in_index_(other.in_index_), in_neighbors_(other.in_neighbors_), is_transpose_(false){
   /* Commenting this because object is not taking owner ship of the elements, notice destructor_free is set to false
        other.num_edges_ = -1;
        other.num_nodes_ = -1;
        other.out_index_ = nullptr;
        other.out_neighbors_ = nullptr;
        other.in_index_ = nullptr;
        other.in_neighbors_ = nullptr;
        other.flags_ = nullptr;
        other.offsets_ = nullptr;
  */
        out_index_shared_ = other.out_index_shared_;
        out_neighbors_shared_ = other.out_neighbors_shared_;
        in_index_shared_ = other.in_index_shared_;
        in_neighbors_shared_ = other.in_neighbors_shared_;
        //Set this up for getting random neighbors
        srand(time(NULL));
	
    }


  CSRGraph(CSRGraph&& other) : directed_(other.directed_),
    num_nodes_(other.num_nodes_), num_edges_(other.num_edges_),
    out_index_(other.out_index_), out_neighbors_(other.out_neighbors_),
    in_index_(other.in_index_), in_neighbors_(other.in_neighbors_), is_transpose_(false){
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.out_index_ = nullptr;
      other.out_neighbors_ = nullptr;
      other.in_index_ = nullptr;
      other.in_neighbors_ = nullptr;
      other.flags_ = nullptr;
    other.offsets_ = nullptr;
       
        out_index_shared_ = other.out_index_shared_;
        out_neighbors_shared_ = other.out_neighbors_shared_;
        in_index_shared_ = other.in_index_shared_;
        in_neighbors_shared_ = other.in_neighbors_shared_;
       
        other.out_index_shared_.reset(); 
        other.out_neighbors_shared_.reset();
        other.in_index_shared_.reset(); 
        other.in_neighbors_shared_.reset();
       
        other.flags_shared_.reset(); 
        other.offsets_shared_.reset();
      //Set this up for getting random neighbors
      srand(time(NULL));
  }


      ~CSRGraph() {
        if (!is_transpose_)
            ReleaseResources();
      }

    CSRGraph& operator=(CSRGraph& other) {
        if (this != &other) {

            if (!is_transpose_)
                ReleaseResources();
            directed_ = other.directed_;
            num_edges_ = other.num_edges_;
            num_nodes_ = other.num_nodes_;
            out_index_ = other.out_index_;
            out_neighbors_ = other.out_neighbors_;
            in_index_ = other.in_index_;
            in_neighbors_ = other.in_neighbors_;
        out_index_shared_ = other.out_index_shared_;
        out_neighbors_shared_ = other.out_neighbors_shared_;
        in_index_shared_ = other.in_index_shared_;
        in_neighbors_shared_ = other.in_neighbors_shared_;
            //need the following, otherwise would get double free errors
/*
          other.num_edges_ = -1;
          other.num_nodes_ = -1;
          other.out_index_ = nullptr;
          other.out_neighbors_ = nullptr;
          other.in_index_ = nullptr;
          other.in_neighbors_ = nullptr;
          other.flags_ = nullptr;
          other.offsets_ = nullptr;
*/
        }
        return *this;
    }

    CSRGraph& operator=(CSRGraph&& other) {
    if (this != &other) {
        if (!is_transpose_ )
            ReleaseResources();
      directed_ = other.directed_;
      num_edges_ = other.num_edges_;
      num_nodes_ = other.num_nodes_;
      out_index_ = other.out_index_;
      out_neighbors_ = other.out_neighbors_;
      in_index_ = other.in_index_;
      in_neighbors_ = other.in_neighbors_;
        out_index_shared_ = other.out_index_shared_;
        out_neighbors_shared_ = other.out_neighbors_shared_;
        in_index_shared_ = other.in_index_shared_;
        in_neighbors_shared_ = other.in_neighbors_shared_;
      other.num_edges_ = -1;
      other.num_nodes_ = -1;
      other.out_index_ = nullptr;
      other.out_neighbors_ = nullptr;
      other.in_index_ = nullptr;
      other.in_neighbors_ = nullptr;
      other.flags_ = nullptr;
      other.offsets_ = nullptr;
        other.out_index_shared_.reset(); 
        other.out_neighbors_shared_.reset();
        other.in_index_shared_.reset(); 
        other.in_neighbors_shared_.reset();
       
        other.flags_shared_.reset(); 
        other.offsets_shared_.reset();
    }
    return *this;
  }

  bool directed() const {
    return directed_;
  }

  int64_t num_nodes() const {
    return num_nodes_;
  }

  int64_t num_edges() const {
    return num_edges_;
  }

  int64_t num_edges_directed() const {
    return directed_ ? num_edges_ : 2*num_edges_;
  }

  int64_t out_degree(NodeID_ v) const {
    return out_index_[v+1] - out_index_[v];
  }

  int64_t in_degree(NodeID_ v) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return in_index_[v+1] - in_index_[v];
  }

  Neighborhood out_neigh(NodeID_ n) const {
    return Neighborhood(n, out_index_);
  }

  Neighborhood in_neigh(NodeID_ n) const {
    static_assert(MakeInverse, "Graph inversion disabled but reading inverse");
    return Neighborhood(n, in_index_);
  }

  NodeID_ get_random_out_neigh(NodeID_ n)  {
      int num_nghs = out_degree(n);
      assert(num_nghs!=0);
      int rand_index = rand() % num_nghs;
      return out_index_[n][rand_index];
  }

  NodeID_ get_random_in_neigh(NodeID_ n)  {
      int num_nghs = in_degree(n);
      assert(num_nghs!=0);
      int rand_index = rand() % num_nghs;
      return in_index_[n][rand_index];
  }

  void PrintStats() const {
    std::cout << "Graph has " << num_nodes_ << " nodes and "
              << num_edges_ << " ";
    if (!directed_)
      std::cout << "un";
    std::cout << "directed edges for degree: ";
    std::cout << num_edges_/num_nodes_ << std::endl;
  }

  void PrintTopology() const {
    for (NodeID_ i=0; i < num_nodes_; i++) {
      std::cout << i << ": ";
      for (DestID_ j : out_neigh(i)) {
        std::cout << j << " ";
      }
      std::cout << std::endl;
    }
  }

  static DestID_** GenIndex(const pvector<SGOffset> &offsets, DestID_* neighs) {
    NodeID_ length = offsets.size();
    DestID_** index = new DestID_*[length];
    #pragma omp parallel for
    for (NodeID_ n=0; n < length; n++)
      index[n] = neighs + offsets[n];
    return index;
  }

  pvector<SGOffset> VertexOffsets(bool in_graph = false) const {
    pvector<SGOffset> offsets(num_nodes_+1);
    for (NodeID_ n=0; n < num_nodes_+1; n++)
      if (in_graph)
        offsets[n] = in_index_[n] - in_index_[0];
      else
        offsets[n] = out_index_[n] - out_index_[0];
    return offsets;
  }

  void SetUpOffsets(bool in_graph = false)  {
      offsets_ = new SGOffset[num_nodes_+1];
      offsets_shared_.reset(offsets_);
      for (NodeID_ n=0; n < num_nodes_+1; n++)
        if (in_graph)
          offsets_[n] = in_index_[n] - in_index_[0];
        else
          offsets_[n] = out_index_[n] - out_index_[0];
    }

  Range<NodeID_> vertices() const {
    return Range<NodeID_>(num_nodes());
  }

  SegmentedGraph<DestID_, NodeID_>* getSegmentedGraph(std::string label, int id) {
    return label_to_segment[label]->getSegmentedGraph(id);
      
  }

  int getNumSegments(std::string label) {
    return label_to_segment[label]->numSegments;      
  }
  
  void buildPullSegmentedGraphs(std::string label, int numSegments, bool numa_aware=false, std::string path="") {
    auto graphSegments = new GraphSegments<DestID_,NodeID_>(numSegments, numa_aware);
    label_to_segment[label] = graphSegments;

#ifdef LOADSEG
    cout << "loading segmented graph from " << path << endl;
#pragma omp parallel for num_threads(numSegments)
    for (int i = 0; i < numSegments; i++) {
      FILE *in;
      in = fopen((path + "/" + std::to_string(i)).c_str(), "r");
      auto sg = graphSegments->getSegmentedGraph(i);
      fread((void *) &sg->numVertices, sizeof(sg->numVertices), 1, in);
      fread((void *) &sg->numEdges, sizeof(sg->numEdges), 1, in);
      sg->allocate(i);
      fread((void *) sg->graphId, sizeof(*sg->graphId), sg->numVertices, in);
      fread((void *) sg->edgeArray, sizeof(*sg->edgeArray), sg->numEdges, in);
      fread((void *) sg->vertexArray, sizeof(*sg->vertexArray), sg->numVertices + 1, in);
      fclose(in);
    }
    return;
#endif
    int segmentRange = (num_nodes() + numSegments - 1) / numSegments;
    //Go through the original graph and count the number of target vertices and edges for each segment
    for (auto d : vertices()){
      for (auto s : in_neigh(d)){
	int segment_id;
	if (std::is_same<DestID_, NodeWeight<>>::value)
	  segment_id = static_cast<NodeWeight<>>(s).v/segmentRange;
	else
	  segment_id = s/segmentRange;
	graphSegments->getSegmentedGraph(segment_id)->countEdge(d);
      }
    }

    //Allocate each segment
    graphSegments->allocate();

    //Add the edges for each segment
    for (auto d : vertices()){
      for (auto s : in_neigh(d)){
	int segment_id;
	if (std::is_same<DestID_, NodeWeight<>>::value)
	  segment_id = static_cast<NodeWeight<>>(s).v/segmentRange;
	else
	  segment_id = s/segmentRange;
	graphSegments->getSegmentedGraph(segment_id)->addEdge(d, s);
      }
    }

#ifdef STORESEG
    cout << "output serialized graph segments to " << path << endl;
#pragma omp parallel for num_threads(numSegments)
    for(int i = 0; i < numSegments; i++) {
      FILE *out = fopen((path + "/" + std::to_string(i)).c_str(), "w");
      auto sg = graphSegments->getSegmentedGraph(i);
      fwrite((void *) &sg->numVertices, sizeof(sg->numVertices), 1, out);
      fwrite((void *) &sg->numEdges, sizeof(sg->numEdges), 1, out);
      fwrite((void *) sg->graphId, sizeof(*sg->graphId), sg->numVertices, out);
      fwrite((void *) sg->edgeArray, sizeof(*sg->edgeArray), sg->numEdges, out);
      fwrite((void *) sg->vertexArray, sizeof(*sg->vertexArray), sg->numVertices + 1, out);
      fclose(out);
    }
#endif
  }

private:
  // Making private so cannot be modified from outside
  //useful for deduplication
  int* flags_;
  SGOffset * offsets_;

  bool is_transpose_;
  bool directed_;
  int64_t num_nodes_;
  int64_t num_edges_;
  DestID_** out_index_;
  DestID_*  out_neighbors_;
  DestID_** in_index_;
  DestID_*  in_neighbors_;

public:
  std::shared_ptr<int> flags_shared_;
  std::shared_ptr<SGOffset> offsets_shared_;

  std::shared_ptr<DestID_*> out_index_shared_;
  std::shared_ptr<DestID_> out_neighbors_shared_;

  std::shared_ptr<DestID_*> in_index_shared_;
  std::shared_ptr<DestID_> in_neighbors_shared_;

  std::map<std::string, GraphSegments<DestID_,NodeID_>*> label_to_segment;
 
  DestID_** get_out_index_(void) {
      return out_index_;
  } 
  DestID_* get_out_neighbors_(void) {
      return out_neighbors_;
  }
  DestID_** get_in_index_(void) {
      return in_index_;
  } 
  DestID_* get_in_neighbors_(void) {
      return in_neighbors_;
  }
  inline int* get_flags_() {
      return flags_;
  }
  inline void set_flags_(int *flags) {
      flags_ = flags;
      flags_shared_.reset(flags);
  }
  inline SGOffset * get_offsets_(void) {
      return offsets_;
  }
};

#endif  // GRAPH_H_
