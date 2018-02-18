#include <math.h>
#include <vector>
#include <assert.h>

using namespace std;

template <class DataT, class Vertex>
struct SegmentedGraph 
{
  int *graphId;
  DataT *edgeArray;
  int *vertexArray;
  int numVertices;
  int numEdges;
  bool allocated;

private:
  int lastLocalIndex;
  Vertex lastVertex;
  Vertex lastEdgeIndex;

public:
  SegmentedGraph()
  {
    allocated = false;
    numVertices = 0;
    numEdges = 0;
    lastVertex = -1;
    lastEdgeIndex = 0;
    lastLocalIndex = 0;
  }

  ~SegmentedGraph()
  {
    delete[] graphId;
    delete[] edgeArray;
    delete[] vertexArray;
  }


  void allocate(int segment_id)
  {

    vertexArray = new int[numVertices + 1]; // start,end of last              
    vertexArray[numVertices] = numEdges;
    edgeArray = new DataT[numEdges];
    graphId = new int[numVertices];
    allocated = true;
  }

  /**
   * Count how many edges we need.
   * @v: dst vertex in pull direction and src vertex in push direction
   **/
  inline
  void countEdge(Vertex v)
  {
    if (v != lastVertex) {
      numVertices++;
      lastVertex = v;
    }
    numEdges++;
  }

  /**
   * Add new edge to each subgraph
   * @v: src in pull direction, dst in push direction
   * @e: dst in pull direction, src in push direction
   **/
  inline
  void addEdge(Vertex toVertexArray, DataT toEdgeArray)
  {
    if (toVertexArray != lastVertex) {
      // a new vertex going to the same partition                                   
      // must be sorted                                                             
      //assert(e > lastVertex);
      lastVertex = toVertexArray;
      graphId[lastLocalIndex] = toVertexArray;
      vertexArray[lastLocalIndex++] = lastEdgeIndex;
    }                                                                               
    edgeArray[lastEdgeIndex++] = toEdgeArray;
  }

  void print(){
    assert(allocated == true);
    cout << "Segmented Graph numVertices: " << numVertices << " numEdges: " << numEdges  << endl;
    cout << "Vertex Array: " << endl;
    for (int i = 0; i < numVertices; i++){
      cout << " " << vertexArray[i];
    }
    cout << endl;

    cout << "Edge Array: " << endl;
    for (int i = 0; i < numEdges; i++){
      cout << " " << edgeArray[i];
    }
    cout << endl;

    cout << "GraphId Array: " << endl;
    for (int i = 0; i < numVertices; i++){
      cout << " " << graphId[i];
    }
    cout << endl;
  }
};

template <class DataT, class Vertex>
struct GraphSegments 
{
  int numSegments;
  vector<SegmentedGraph<DataT,Vertex>*> segments;
  
  GraphSegments(int _numSegments): numSegments(_numSegments)
  {
    //alocate each graph segment
    for (int i=0; i<numSegments; i++){
      segments.push_back(new SegmentedGraph<DataT, Vertex>());
    }
  }

  ~GraphSegments(){
    //deallocate every graph segment
   for (int i=0; i<numSegments; i++){
     delete segments[i];
   }
  }

  void allocate() {
    for (int i = 0; i<numSegments; i++){
      segments[i]->allocate(i);
    }
  }

  SegmentedGraph<DataT, Vertex> * getSegmentedGraph(int id){
    return segments[id];
  }
};

template <class DataT, class Vertex>
void BuildPullSegmentedGraphsUnweighted(const Graph* originalGraph, GraphSegments<DataT,Vertex> * graphSegments, int segmentRange)
{
  //Go through the original graph and count the number of target vertices and edges for each segment
  for (NodeID v : originalGraph->vertices()){
    for (NodeID u : originalGraph->in_neigh(v)){
      int segment_id = u/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->countEdge(v);
    }
  }

  //Allocate each segment
  graphSegments->allocate();

  //Add the edges for each segment
  for (NodeID v : originalGraph->vertices()){
    for (NodeID u : originalGraph->in_neigh(v)){
      int segment_id = u/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->addEdge(v, u);
    }
  }
}

template <class DataT, class Vertex>
void BuildPullSegmentedGraphsWeighted(const WGraph* originalGraph, GraphSegments<DataT,Vertex> * graphSegments, int segmentRange)
{
  //Go through the original graph and count the number of target vertices and edges for each segment
  for (NodeID v : originalGraph->vertices()){
    for (WNode u : originalGraph->in_neigh(v)){
      int segment_id = u.v/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->countEdge(v);
    }
  }

  //Allocate each segment
  graphSegments->allocate();

  //Add the edges for each segment
  for (NodeID v : originalGraph->vertices()){
    for (WNode u : originalGraph->in_neigh(v)){
      int segment_id = u.v/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->addEdge(v, u);
    }
  }
}

template <class DataT, class Vertex>
void BuildPushSegmentedGraphs(const Graph* originalGraph, GraphSegments<DataT,Vertex> * graphSegments, int segmentRange)
{
  //Go through the original graph and count the number of target vertices and edges for each segment
  for (NodeID u : originalGraph->vertices()){
    for (NodeID v : originalGraph->out_neigh(u)){
      int segment_id = v/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->countEdge(u);
    }
  }

  //Allocate each segment
  graphSegments->allocate();

  //Add the edges for each segment
  for (NodeID u : originalGraph->vertices()){
    for (NodeID v : originalGraph->out_neigh(u)){
      int segment_id = v/segmentRange;
      graphSegments->getSegmentedGraph(segment_id)->addEdge(u, v);
    }
  }
}

