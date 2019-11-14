#include "intrinsics.h"
extern int  * __restrict D;
extern Graph edges;
extern julienne::PriorityQueue < int  >* pq;
constexpr julienne::uintE COVERED = ((julienne::uintE)INT_E_MAX) - 1;  
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
//auto ExternFunction = [&] (julienne::vertexSubset active) -> julienne::vertexSubset {


constexpr double epsilon = 0.01;
const double x = 1.0/log(1.0 + epsilon);


julienne::vertexSubset extern_function(julienne::vertexSubset active) {
    static julienne::array_imap<julienne::uintE> *Elms_p = NULL;
    if (Elms_p == NULL)
         Elms_p = new julienne::array_imap<julienne::uintE>(edges.julienne_graph.n, [&] (size_t i) { return UINT_E_MAX; });

    auto &Elms = *Elms_p;
    auto &G = edges.julienne_graph;

    auto get_bucket_clamped = [&] (size_t deg) -> julienne::uintE { return (deg == 0) ? UINT_E_MAX : (julienne::uintE)floor(x * log((double) deg)); };
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
int get_cover_size(void) {
	return cover.size;
}
/* Extern C++ code ends */
