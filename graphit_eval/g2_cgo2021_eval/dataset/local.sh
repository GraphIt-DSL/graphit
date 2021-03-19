if [[ $(hostname) == "lanka-dgx0.csail.mit.edu" ]]; then
	echo OK
	ln -s /local/ajaybr/graph-dataset/clean_general/*.mtx .
else
	echo You are not running this command on the right host. Please use `make dataset` instead
fi
