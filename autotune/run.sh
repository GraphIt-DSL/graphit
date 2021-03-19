
export CUDA_VISIBLE_DEVICES=6


#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-LiveJournal1.mtx --algo_file gpu_apps/sssp_delta_stepping.gt --killed_process_report_runtime_limit 1 --max_delta 100 --runtime_limit 20 --stop-after 600 --final_config=final_config_ds_livejournal.json --kernel_fusion=True --num_vertices=0
#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-twitter-2010.mtx --algo_file gpu_apps/sssp_delta_stepping.gt --killed_process_report_runtime_limit 1 --max_delta 100 --runtime_limit 20 --stop-after 600 --final_config=final_config_ds_twitter.json --kernel_fusion=True --num_vertices=0
#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/road_usa.weighted.mtx --algo_file gpu_apps/sssp_delta_stepping.gt --killed_process_report_runtime_limit 1 --max_delta 100000 --runtime_limit 20 --stop-after 1500 --final_config=final_config_ds_road_usa.json --kernel_fusion=True --num_vertices=0

#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-LiveJournal1.mtx --algo_file gpu_apps/cc.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 10 --stop-after 600 --final_config=final_config_cc_livejournal.json --num_vertices=0
#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-twitter-2010.mtx --algo_file gpu_apps/cc.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 30 --stop-after 600 --final_config=final_config_cc_twitter.json --num_vertices=0

#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/road_usa.weighted.mtx --algo_file gpu_apps/cc.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after 600 --final_config=final_config_cc_road_usa.json --num_vertices=0

#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-LiveJournal1.mtx --algo_file gpu_apps/pagerank.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after 600 --final_config=final_config_pr_livejournal.json --kernel_fusion=True --edge_only=True --num_vertices=4847571
#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-twitter-2010.mtx --algo_file gpu_apps/pagerank.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 30 --stop-after 600 --final_config=final_config_pr_twitter.json --kernel_fusion=True --edge_only=True --num_vertices=21297772
#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/road_usa.weighted.mtx --algo_file gpu_apps/pagerank.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after 600 --final_config=final_config_pr_road_usa.json --kernel_fusion=True --edge_only=True --num_vertices=23947347



python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-orkut.mtx --algo_file gpu_apps/bfs.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after=36000  --final_config=final_config_bfs_orkut.json --kernel_fusion=True --num_vertices=0 --hybrid_schedule=True

python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-LiveJournal1.mtx --algo_file gpu_apps/bfs.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after=3600  --final_config=final_config_bfs_livejournal.json --kernel_fusion=True --num_vertices=0 --hybrid_schedule=True --hybrid_threshold=8


#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/soc-twitter-2010.mtx --algo_file gpu_apps/bfs.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after 600 --final_config=final_config_bfs_twitter.json --kernel_fusion=True --num_vertices=0 --hybrid_schedule=True

#python3 graphit_gpu_autotuner.py --graph /local/ajaybr/graph-dataset/clean_general/road_usa.weighted.mtx --algo_file gpu_apps/bfs.gt --killed_process_report_runtime_limit 1 --max_delta 1 --runtime_limit 20 --stop-after 600 --final_config=final_config_bfs_road_usa.json --kernel_fusion=True --num_vertices=0 --hybrid_schedule=1

