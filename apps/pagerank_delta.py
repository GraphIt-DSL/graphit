import graphit
import scipy.io
from scipy.sparse import csr_matrix
import sys
import time

module = graphit.compile_and_load("pagerank_delta_export.gt")
graph = csr_matrix(scipy.io.mmread(sys.argv[1]))
module.set_graph(graph)
start_time = time.perf_counter()
ranks = module.do_pagerank_delta()
end_time = time.perf_counter()

print ("Time elapsed = " + str(end_time - start_time) + " seconds")
