# This file is copied from a repo - (https://github.com/bespoke-silicon-group/hb_starlite/tree/master/py-graphit-example)
# author: mrutt92
 
import graphit
from scipy.sparse import csr_matrix


# In GraphIt, `edgeset`s define graphs. Use SciPy's `csr_matrix` to
# create these values, which we'll pass into a GraphIt function.
graph = csr_matrix((
    [4, 5, 6, 4, 5, 6],
    [1, 2, 3, 0, 0, 0],
    [0, 3, 4, 5, 6],
))

# Compile a GraphIt source file and load it so we can call its function.
sssp_module = graphit.compile_and_load("./sssp.gt")

# Invoke the `do_sssp` GraphIt function from the loaded module.
distances = sssp_module.do_sssp(graph, 0)

# A vector{Vertex} value is represented as a NumPy array. Use
# `torch.tensor(...)` to convert it to a tensor.
print(distances)