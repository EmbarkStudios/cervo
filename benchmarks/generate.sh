if [ "$#" -lt 1 ] ; then
	echo "usage: ./generate.sh SUFFIX"
	exit 1
fi

suffix="$1"

count=1000
if [ "$#" -ge 2 ] ; then
	count=$2
fi



if ! python python/check_deps.py ; then
	exit 1
fi

cargo build --release --bin perf-test || exit 1

mkdir -p out
cargo run --release --bin perf-test batchers $count 20 "out/batchers-$suffix.csv" -o ../brains/test-large.onnx -n ../brains/test-large.nnef.tar -f 1,6
python python/batchers.py "out/batchers-$suffix.csv" 20 "out/batchers-$suffix.png"

cargo run --release --bin perf-test loading 100 "out/loaders-$suffix.csv" -o ../brains/test-large.onnx -n ../brains/test-large.nnef.tar
python python/loaders.py "out/loaders-$suffix.csv" 100 "out/loaders-$suffix.png"
