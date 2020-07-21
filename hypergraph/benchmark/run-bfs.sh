inp="/data/scratch/rlhuang/inputs"
winp="$inp-wgh"
hp="./HyperBFS"
gp="hyperbfs"
graph=$1
export CILK=1
cd ~/ppopp20-ae/apps/hyper
make clean &> /dev/null
graphinfo=($(head -5 "$inp/$graph" | tr "\n" "\n"))
echo "CILK"
echo $graph
head -5 "$inp/$graph"
echo "PROOF: Graph $graph has v=${graphinfo[1]}, h=${graphinfo[3]}, mv=${graphinfo[2]}, mh=${graphinfo[4]}"
for suff in "" "-Sparse" "-Dense" "-Threshold" "-EdgeAware" "-Sparse-EdgeAware" "-Dense-EdgeAware" "-Threshold-EdgeAware"; do
    make "$hp$suff" &> /dev/null
    echo "$hp$suff"
    for n in {1,2,4,8,16,32,48}; do
	echo "threads=$n"
	CILK_NWORKERS=$n "$hp$suff" -rounds 3 "$inp/$graph"
    done
    echo "threads=99"
        numactl -i all "$hp$suff" -rounds 3 "$inp/$graph"
    echo "threads=999"
done
cd ~/graphit/build/bin/
echo "graphit start"
for sched in $(ls ../../hypergraph/schedules/same/*.gt); do
    echo $sched
    python3 graphitc.py -a "../../apps/$gp.gt" -f "$sched" -o "$gp.cpp"
    g++ -std=c++14 -I ../../src/runtime_lib/ -DCILK -fcilkplus -lcilkrts -O3 "$gp.cpp" -o "$gp" &> /dev/null
    for n in {1,2,4,8,16,32,48}; do
	echo "threads=$n"
	CILK_NWORKERS=$n "./$gp" "$inp/$graph.sg" 0 ${graphinfo[1]} ${graphinfo[3]}
    done
    echo "threads=99"
        numactl -i all "./$gp" "$inp/$graph.sg" 0 ${graphinfo[1]} ${graphinfo[3]}
    echo "threads=999"
done
