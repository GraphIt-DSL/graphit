#include "swarm_intrinsics.h"
#include "scc/queues.h"
#include "scc/autoparallel.h"
int __argc;
char **__argv;
swarm_runtime::GraphT<int> edges;
int *dist;
swarm::BucketQueue<int> swarm_pq;
int swarm_pq_delta;
void dist_generated_vector_op_apply_func_0(int v) {
	dist[v] = 2147483647;
}
bool updateEdge(int src, int dst, int weight) {
	bool output2;
	bool dist_trackving_var_1 = (bool)0;
	int new_dist = (dist[src] + weight);
	if (dist[dst] > new_dist) {
		dist[dst] = new_dist;
	};
	output2 = dist_trackving_var_1;
	return output2;
}
void printDist(int v) {
	swarm_runtime::print(dist[v]);
}
void reset(int v) {
	dist[v] = 2147483647;
}
SWARM_FUNC_ATTRIBUTES
void swarm_main() {
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		reset(_iter);
	};
	int start_vertex = atoi(__argv[2]);
	dist[start_vertex] = 0;
	swarm_pq_delta = 1;
	swarm_pq.push_init(0, start_vertex);
	swarm_pq.for_each_prio([](unsigned level, int src, auto push) {
		int32_t edgeZero = edges.h_src_offsets[src];
		int32_t edgeLast = edges.h_src_offsets[src+1];
		for (int i = edgeZero; i < edgeLast; i++) {
			int dst = edges.h_edge_dst[i];
			int weight = edges.h_edge_weight[i];
			{
				bool output2;
				bool dist_trackving_var_1 = (bool)0;
				int new_dist = (dist[src] + weight);
				if (dist[dst] > new_dist) {
					dist[dst] = new_dist;
					push((new_dist)/swarm_pq_delta, dst);
				};
				output2 = dist_trackving_var_1;
			}
		}
	}, [](unsigned level, int src) {
	});
}
#include <queue>
#include <utility>
// Compares against simple serial implementation
bool verify() {
    int start_vertex = atoi(__argv[2]);
    // Serial Dijkstra's algorithm implementation to get oracle distances
    int* oracle_dist = new int[edges.num_vertices];
    for (int _iter = 0; _iter < edges.num_vertices; _iter++) {
        oracle_dist[_iter] = 2147483647;
    };
    using entry = std::pair<int, int>;
    using namespace std;
    std::priority_queue<entry, std::vector<entry>, std::greater<entry>> mq;
    mq.push(make_pair(0, start_vertex));
    while (!mq.empty()) {
        int td = mq.top().first;
        int u = mq.top().second;
        mq.pop();
        //printf("checking node %d at tentative distance %d\n", u, td);
        if (oracle_dist[u] == 2147483647) {
            //printf("setting node %d to distance %d\n", u, td);
            oracle_dist[u] = td;
            for (int eid = edges.h_src_offsets[u]; eid < edges.h_src_offsets[u+1]; eid++) {
                int neigh = edges.h_edge_dst[eid];
                int weight = edges.h_edge_weight[eid];
                int new_dist = td + weight;
                //printf("pushing node %d at tentative distance %d\n", neigh, new_dist);
                mq.push(std::make_pair(new_dist, neigh));
            }
        }
    }
    // Report any mismatches
    bool all_ok = true;
    for (int n = 0; n < edges.num_vertices; n++) {
        if (dist[n] != oracle_dist[n]) {
            cout << n << ": " << dist[n] << " != " << oracle_dist[n] << endl;
            all_ok = false;
        }
    }
    return all_ok;
}

int main(int argc, char* argv[]) {
	__argc = argc;
	__argv = argv;
	swarm_runtime::load_graph(edges, __argv[1]);
	dist = new int[swarm_runtime::builtin_getVertices(edges)];
	for (int _iter = 0; _iter < swarm_runtime::builtin_getVertices(edges); _iter++) {
		dist_generated_vector_op_apply_func_0(_iter);
	};
	SCC_PARALLEL( swarm_main(); );

	if (verify())
                std::cout << "SUCCESS!!!!!!!!!!!!!!" << std::endl;
        else
                std::cout << "FAILED!!!!!!!!!!!!!!!" << std::endl;
}
