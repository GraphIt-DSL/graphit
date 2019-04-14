//
// Created by Yunming Zhang on 4/27/17.
//

#include <gtest.h>
#include "intrinsics.h"
#include "infra_gapbs/graph_verifier.h"





class RuntimeLibTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }

    virtual void TearDown() {

    }


    // Compares against simple serial implementation
    bool SSSPVerifier(const WGraph &g, NodeID source,
                      const pvector<WeightT> &dist_to_test) {



        // Serial Dijkstra implementation to get oracle distances
        pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
        oracle_dist[source] = 0;
        typedef pair<WeightT, NodeID> WN;
        priority_queue<WN, vector<WN>, greater<WN>> mq;
        mq.push(make_pair(0, source));
        while (!mq.empty()) {
            WeightT td = mq.top().first;
            NodeID u = mq.top().second;
            mq.pop();
            if (td == oracle_dist[u]) {
                for (WNode wn : g.out_neigh(u)) {
                    if (td + wn.w < oracle_dist[wn.v]) {
                        oracle_dist[wn.v] = td + wn.w;
                        mq.push(make_pair(td + wn.w, wn.v));
                    }
                }
            }
        }
        // Report any mismatches
        bool all_ok = true;
        for (NodeID n : g.vertices()) {
            if (dist_to_test[n] != oracle_dist[n]) {
                cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
                all_ok = false;
            }
        }
        return all_ok;
    }



    // Compares against simple serial implementation
    bool PPSPVerifier(const WGraph &g, NodeID source, NodeID dest,
                      const pvector<WeightT> &dist_to_test) {
        // Serial Dijkstra implementation to get oracle distances
        pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
        oracle_dist[source] = 0;
        typedef pair<WeightT, NodeID> WN;
        priority_queue<WN, vector<WN>, greater<WN>> mq;
        mq.push(make_pair(0, source));
        while (!mq.empty()) {
            WeightT td = mq.top().first;
            NodeID u = mq.top().second;
            mq.pop();
            if (td == oracle_dist[u]) {
                for (WNode wn : g.out_neigh(u)) {
                    if (td + wn.w < oracle_dist[wn.v]) {
                        oracle_dist[wn.v] = td + wn.w;
                        mq.push(make_pair(td + wn.w, wn.v));
                    }
                }
            }
        }


        // Report any mismatches
        //bool all_ok = true;
        //for (NodeID n : g.vertices()) {
        //if (dist_to_test[n] != oracle_dist[n]) {
        //    cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
        //    all_ok = false;
        //  }
        //}

        bool all_ok = false;
        if (dist_to_test[dest] == oracle_dist[dest]) all_ok = true;
        else {
            cout << "measured dist: " << dist_to_test[dest] << endl;
            cout << "oracle dist: " << oracle_dist[dest] << endl;
        }

        return all_ok;
    }


    const WeightT kDistInf = std::numeric_limits<WeightT>::max() / 2;

};


TEST_F(RuntimeLibTest, SimpleLoadGraphFromFileTest) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    EXPECT_EQ (7 , g.num_edges());
}

TEST_F(RuntimeLibTest, SimpleLoadVerticesromEdges) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    int num_vertices = builtin_getVertices(g);
    //max node id + 1, assumes the first node has id 0
    EXPECT_EQ (5 , num_vertices);
}

TEST_F(RuntimeLibTest, GetOutDegrees) {
    Graph g = builtin_loadEdgesFromFile("../../test/graphs/test.el");
    auto out_degrees = builtin_getOutDegrees(g);
    //max node id + 1, assumes the first node has id 0
    //TODO: fix this, we don't use vectors anymore
    //EXPECT_EQ (5 , out_degrees.size());
}

TEST_F(RuntimeLibTest, TimerTest) {
//    float start_time = getTime();
//    sleep(1);
//    float end_time = getTime();
    startTimer();
    sleep(1);
    float elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;

    startTimer();
    sleep(2);
    elapsed_time = stopTimer();
    std::cout << "elapsed_time: " << elapsed_time << std::endl;
    EXPECT_EQ (5 , 5);
}

TEST_F(RuntimeLibTest, VertexSubsetSimpleTest) {
    bool test_flag = true;
    auto vertexSubset = new VertexSubset<int>(5, 0);
    for (int v = 0; v < 5; v = v+2){
        vertexSubset->addVertex(v);
    }

    for (int v = 1; v < 5; v = v+2){
        if (vertexSubset->contains(v))
            test_flag = false;
    }

    for (int v = 0; v < 5; v = v+2){
        if (!vertexSubset->contains(v))
            test_flag = false;
    }

    EXPECT_EQ(builtin_getVertexSetSize(vertexSubset), 3);


    delete vertexSubset;

    EXPECT_EQ(true, test_flag);

}


//test init of the eager priority queue based on GAPBS
TEST_F(RuntimeLibTest, EagerPriorityQueueInit) {

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.wel");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }
    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array);

    EXPECT_EQ(pq.get_current_priority(), 0);
}

//test init of the buffered priority queue based on Julienne
TEST_F(RuntimeLibTest, BufferedPriorityQueueInit) {


    EXPECT_EQ(true, true);
}

// test compilation of the C++ version of SSSP using eager priority queue
TEST_F(RuntimeLibTest, SSSPOrderProcessingWithMergeTest){

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.wel");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }

    NodeID source = 2;
    dist_array[source] = 0;

    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array);
    WeightT input_delta = 2;

    auto while_cond_func = [&]()->bool{
            return !pq.finished();
    };


    auto edge_update_func = [&](vector<vector<NodeID> >& local_bins, NodeID src, NodeID dst, WeightT wt) {
        WeightT old_dist = dist_array[dst];
        WeightT new_dist = dist_array[src] + wt;
        updatePriorityMin<WeightT>()(&pq, local_bins, dst, old_dist, new_dist);
    };

    OrderedProcessingOperatorWithMerge(&pq, g, while_cond_func, edge_update_func, 1000,  source);

    pvector<WeightT> dist = pvector<WeightT>(g.num_nodes());
    for (int i = 0; i < g.num_nodes(); i++){
        dist[i]= dist_array[i];
    }

    EXPECT_EQ(SSSPVerifier(g, source, dist), true);
}


// test compilation of the C++ version of SSSP using eager priority queue
TEST_F(RuntimeLibTest, SSSPOrderProcessingNoMergeTest){

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/4.wel");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }

    NodeID source = 0;
    dist_array[source] = 0;
    int delta = 3;
    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array, delta);

    auto while_cond_func = [&]()->bool{
        return !pq.finished();
    };


    auto edge_update_func = [&](vector<vector<NodeID> >& local_bins, NodeID src, NodeID dst, WeightT wt) {
        WeightT old_dist = dist_array[dst];
        WeightT new_dist = dist_array[src] + wt;
        updatePriorityMin<WeightT>()(&pq, local_bins, dst, old_dist, new_dist);
    };

    OrderedProcessingOperatorNoMerge(&pq, g,  while_cond_func, edge_update_func,  source);

    pvector<WeightT> dist = pvector<WeightT>(g.num_nodes());
    for (int i = 0; i < g.num_nodes(); i++){
        dist[i]= dist_array[i];
    }

    EXPECT_EQ(SSSPVerifier(g, source, dist), true);
}

// test compilation of the C++ version of PPSP using eager priority queue
TEST_F(RuntimeLibTest, PPSPOrderProcessingNoMergeTest){

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.wel");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }

    NodeID source = 0;
    NodeID dest = 3;
    dist_array[source] = 0;

    WeightT input_delta = 3;
    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array, input_delta);

    auto while_cond_func = [&]()->bool{
            return !pq.finishedNode(dest);
    };


    auto edge_update_func = [&](vector<vector<NodeID> >& local_bins, NodeID src, NodeID dst, WeightT wt) {
        WeightT old_dist = dist_array[dst];
        WeightT new_dist = dist_array[src] + wt;
        updatePriorityMin<WeightT>()(&pq, local_bins, dst, old_dist, new_dist);
    };

    OrderedProcessingOperatorNoMerge(&pq, g, [&]()->bool{return !pq.finishedNode(dest); },
            edge_update_func,  source);

    pvector<WeightT> dist = pvector<WeightT>(g.num_nodes());
    for (int i = 0; i < g.num_nodes(); i++){
        dist[i]= dist_array[i];
    }

    delete[] dist_array;

    EXPECT_EQ(PPSPVerifier(g, source, dest, dist), true);
}


TEST_F(RuntimeLibTest, PPSPOrderProcessingWithMergeTest){

    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/test.wel");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    for (int i = 0; i < g.num_nodes(); i++){
        dist_array[i] = kDistInf;
    }

    NodeID source = 0;
    NodeID dest = 3;
    dist_array[source] = 0;

    WeightT input_delta = 2;
    EagerPriorityQueue<WeightT> pq = EagerPriorityQueue<WeightT>(dist_array, input_delta);

    auto while_cond_func = [&]()->bool{
        return !pq.finishedNode(dest);
    };


    auto edge_update_func = [&](vector<vector<NodeID> >& local_bins, NodeID src, NodeID dst, WeightT wt) {
        WeightT old_dist = dist_array[dst];
        WeightT new_dist = dist_array[src] + wt;
        updatePriorityMin<WeightT>()(&pq, local_bins, dst, old_dist, new_dist);
    };

    OrderedProcessingOperatorWithMerge(&pq, g, while_cond_func, edge_update_func, 1000, source);

    pvector<WeightT> dist = pvector<WeightT>(g.num_nodes());
    for (int i = 0; i < g.num_nodes(); i++){
        dist[i]= dist_array[i];
    }

    delete[] dist_array;

    EXPECT_EQ(PPSPVerifier(g, source, dest, dist), true);
}

TEST_F(RuntimeLibTest, AStar_load_graph){
    WGraph g = builtin_loadWeightedEdgesFromFile("../../test/graphs/monaco.bin");
    WeightT* dist_array = new WeightT[g.num_nodes()];
    std::cout << "number of nodes: " << g.num_nodes() << std::endl;

}

// test compilation of the C++ version of KCore using buffered priority queue
TEST_F(RuntimeLibTest, KCore_test){
    char iFile[] = "../../test/graphs/rMatGraph_J_5_100";
    bool symmetric = true;
    bool compressed = false;
    bool binary = false;
    bool mmap = false;
    julienne::graph<julienne::symmetricVertex> G = julienne::readGraph<julienne::symmetricVertex>(iFile, compressed, symmetric, binary, mmap);
    
    // Compute

    auto &GA = G;
    const size_t n = GA.n; 
    const size_t m = GA.m;
    julienne::uintE* updated_degree;
    updated_degree = new julienne::uintE[n];
    
    parallel_for(int i = 0; i < n; i++) updated_degree[i] = GA.V[i].getOutDegree();
    size_t num_buckets = 128;
    auto pq = new julienne::PriorityQueue<julienne::uintE>(n, updated_degree, julienne::increasing, julienne::strictly_decreasing, num_buckets);

    auto apply_function = [&] (const tuple<julienne::uintE, julienne::uintE>& p) {
        julienne::uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
        julienne::uintE deg = updated_degree[v];
        julienne::uintE k = pq->get_current_priority();
        if (deg > k) {
            julienne::uintE new_deg = std::max(deg - edgesRemoved, k);
            updated_degree[v] = new_deg;
            julienne::uintE bkt = pq->get_bucket(new_deg);
            return julienne::wrap(v, bkt);
        }
        return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE> >();
    };


    auto em = julienne::EdgeMap<julienne::uintE, julienne::symmetricVertex>(GA, std::make_tuple(UINT_E_MAX, 0), (size_t)GA.m/5);
    size_t finished = 0;
    while (finished != n) {
        auto active = pq->next_bucket().identifiers;
        finished += active.size();
        julienne::vertexSubsetData<julienne::uintE> moved = em.edgeMapCount<julienne::uintE>(active, apply_function);
        pq->update_buckets(moved.get_fn_repr(), moved.size());
        moved.del();
        active.del();
    }

    delete pq;

    julienne::uintE mc = 0;
    for (size_t i=0; i < GA.n; i++) { 
        mc = std::max(mc, updated_degree[i]); 
    }
    EXPECT_EQ(4, mc);
    delete[] updated_degree;
    G.del();
}



TEST_F(RuntimeLibTest, SetCover_test){
    char iFile[] = "../../test/graphs/rMatGraph_J_5_100";
    bool symmetric = true;
    bool compressed = false;
    bool binary = false;
    bool mmap = false;
    julienne::graph<julienne::symmetricVertex> G = julienne::readGraph<julienne::symmetricVertex>(iFile, compressed, symmetric, binary, mmap);
    
    // Compute
    auto GA = G;
    //auto D = julienne::array_imap<julienne::uintE>(G.n, [&] (size_t i) { return G.V[i].getOutDegree(); });
    constexpr double epsilon = 0.01;
    const double x = 1.0/log(1.0 + epsilon);

    auto get_bucket_clamped = [&] (size_t deg) -> julienne::uintE { return (deg == 0) ? UINT_E_MAX : (julienne::uintE)floor(x * log((double) deg)); };
    auto D = new julienne::uintE[G.n];
    parallel_for (int index = 0; index < G.n; index++)
        D[index] = get_bucket_clamped(G.V[index].getOutDegree()); 

    constexpr julienne::uintE COVERED = ((julienne::uintE)INT_E_MAX) - 1;
    auto max_f = [] (julienne::uintE x, julienne::uintE y) { return std::max(x,y); };
    
    size_t num_buckets = 128;
    auto pq = new julienne::PriorityQueue<julienne::uintE>(G.n, D, julienne::decreasing, julienne::strictly_decreasing, num_buckets);


    /* Extern C++ code start */
    struct Visit_Elms {
        julienne::uintE* elms;
        Visit_Elms(julienne::uintE* _elms) : elms(_elms) { }
        inline bool updateAtomic(const julienne::uintE& s, const julienne::uintE& d) {
            julienne::uintE oval = elms[d];
            julienne::writeMin(&(elms[d]), s);
            return false;
        }
        inline bool update(const julienne::uintE& s, const julienne::uintE& d) { return updateAtomic(s, d); }
        inline bool cond(const julienne::uintE& d) const { return elms[d] != COVERED; }
    };
    julienne::dyn_arr<julienne::uintE> cover = julienne::dyn_arr<julienne::uintE>();
    auto Elms = julienne::array_imap<julienne::uintE>(G.n, [&] (size_t i) { return UINT_E_MAX; });
    auto ExternFunction = [&] (julienne::vertexSubset active) -> julienne::vertexSubset {
        // 1. sets -> elements (Pack out sets and update their degree)
        auto pack_predicate = [&] (const julienne::uintE& u, const julienne::uintE& ngh) { return Elms[ngh] != COVERED; };
        auto pack_apply = [&] (julienne::uintE v, size_t ct) { D[v] = get_bucket_clamped(ct); };
        auto packed_vtxs = julienne::edgeMapFilter(G, active, pack_predicate, julienne::pack_edges);
        julienne::vertexMap(packed_vtxs, pack_apply);
        // Calculate the sets which still have sufficient degree (degree >= threshold)
        size_t threshold = ceil(pow(1.0+epsilon, pq->get_current_priority()));
        auto above_threshold = [&] (const julienne::uintE& v, const julienne::uintE& deg) { return deg >= threshold; };
        auto still_active = julienne::vertexFilter2<julienne::uintE>(packed_vtxs, above_threshold);
        packed_vtxs.del();
        // 2. sets -> elements (writeMin to acquire neighboring elements)
        julienne::edgeMap(G, still_active, Visit_Elms(Elms.s), -1, julienne::no_output | julienne::dense_forward);
        // 3. sets -> elements (count and add to cover if enough elms were won)
        const size_t low_threshold = std::max((size_t)ceil(pow(1.0+epsilon,pq->get_current_priority()-1)), (size_t)1);
        auto won_ngh_f = [&] (const julienne::uintE& u, const julienne::uintE& v) -> bool { return Elms[v] == u; };
        auto threshold_f = [&] (const julienne::uintE& v, const julienne::uintE& numWon) {
          if (numWon >= low_threshold) D[v] = UINT_E_MAX;
        };
        auto activeAndCts = julienne::edgeMapFilter(G, still_active, won_ngh_f);
        julienne::vertexMap(activeAndCts, threshold_f);
        auto inCover = julienne::vertexFilter2(activeAndCts, [&] (const julienne::uintE& v, const julienne::uintE& numWon) {
            return numWon >= low_threshold; });
        cover.copyInF([&] (julienne::uintE i) { return inCover.vtx(i); }, inCover.size());
        inCover.del(); activeAndCts.del();
        // 4. sets -> elements (Sets that joined the cover mark their neighboring
        // elements as covered. Sets that didn't reset any acquired elements)
        auto reset_f = [&] (const julienne::uintE& u, const julienne::uintE& v) -> bool {
          if (Elms[v] == u) {
            if (D[u] == UINT_E_MAX) Elms[v] = COVERED;
            else Elms[v] = UINT_E_MAX;
          } return false;
        };
        julienne::edgeMap(G, still_active, julienne::EdgeMap_F<decltype(reset_f)>(reset_f), -1, julienne::no_output | julienne::dense_forward);
        still_active.del();
        return active;
    }; 

    /* Extern C++ code ends */
    int rounds = 0;
    while (1) {
        auto bucket = pq->next_bucket();
        if (pq->finished()) {
            break;
        }
        auto active = bucket.identifiers;
        auto modified = ExternFunction(active);
        modified.toSparse();
        auto f = [&] (size_t i) -> julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>> {
            const julienne::uintE v = modified.vtx(i);
            const julienne::uintE bkt = pq->get_bucket(D[v]);
            return julienne::Maybe<std::tuple<julienne::uintE, julienne::uintE>>(std::make_tuple(v, bkt));
        };
        pq->update_buckets(f, modified.size());
        
        active.del();    
        rounds+=1;

    }  
    
    EXPECT_EQ(rounds, 24);
    EXPECT_EQ(cover.size, 33); 

    cover.del();
 
    G.del();

}
