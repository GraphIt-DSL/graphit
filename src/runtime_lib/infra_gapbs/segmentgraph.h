#include <math.h>
#include <vector>
#include <assert.h>
#ifdef NUMA
#include <omp.h>
#include <numa.h>
#endif

using namespace std;

template <class DataT, class Vertex>
struct SegmentedGraph 
{
  int *graphId;
  DataT *edgeArray;
  int64_t *vertexArray;
  int64_t numVertices;
  int64_t numEdges;
  bool allocated;
  bool numa_aware;

private:
  int64_t lastLocalIndex;
  Vertex lastVertex;
  int64_t lastEdgeIndex;

public:
  SegmentedGraph(bool numa_aware_) : numa_aware(numa_aware_)
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
#ifdef NUMA
    if (numa_aware) {
      numa_free(graphId, sizeof(int) * numVertices);
      numa_free(edgeArray, sizeof(int) * numEdges);
      numa_free(vertexArray, sizeof(int64_t) * (numVertices + 1));
      return;
    }
#endif
    delete[] graphId;
    delete[] edgeArray;
    delete[] vertexArray;
  }


  void allocate(int segment_id)
  {
#ifdef NUMA
    if (numa_aware) {
      int place_id = segment_id % omp_get_num_places();
      vertexArray = (int64_t *)numa_alloc_onnode(sizeof(int64_t) * (numVertices + 1), place_id);
      edgeArray = (DataT *)numa_alloc_onnode(sizeof(DataT) * numEdges, place_id);
      graphId = (int *)numa_alloc_onnode(sizeof(int) * numVertices, place_id);
      vertexArray[numVertices] = numEdges;
      allocated = true;
      lastVertex = -1; // reset lastVertex which is used to point to the dst vertex of the last edge added
      return;
    }
#endif
    vertexArray = new int64_t[numVertices + 1]; // start,end of last              
    edgeArray = new DataT[numEdges];
    graphId = new int[numVertices];
    vertexArray[numVertices] = numEdges;
    allocated = true;
    lastVertex = -1; // reset lastVertex which is used to point to the dst vertex of the last edge added
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
    for (int i = 0; i < numVertices + 1; i++){
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
  
GraphSegments(int _numSegments, bool numa_aware): numSegments(_numSegments)
  {
    //alocate each graph segment
    for (int i=0; i<numSegments; i++){
      segments.push_back(new SegmentedGraph<DataT, Vertex>(numa_aware));
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
