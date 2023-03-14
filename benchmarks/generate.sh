set -eux

if [ "$#" -lt 1 ] ; then
	echo "usage: ./generate.sh SUFFIX"
	exit 1
fi

suffix="$1"

count=10000
if [ "$#" -ge 2 ] ; then
	count=$2
fi


if ! python3 ../python/check_deps.py ; then
	exit 1
fi

FLAGS=--release

cargo build $FLAGS || exit 1

mkdir -p ../out
# cargo run $FLAGS batchers $count 20 "../out/batchers-$suffix.csv" -o ../../brains/test-large.onnx -n ../../brains/test-large.nnef.tar -f 1,6
# python3 ../python/batchers.py "../out/batchers-$suffix.csv" 20 "../out/batchers-$suffix.png"

# cargo run $FLAGS loading $count "../out/loaders-$suffix.csv" -o ../../brains/test-large.onnx -n ../../brains/test-large.nnef.tar
# python3 ../python/loaders.py "../out/loaders-$suffix.csv" $count "../out/loaders-$suffix.png"

cargo run $FLAGS batch-scaling $count "../out/batchsize-$suffix.csv" -o ../../brains/test-large.onnx -b `seq -s, 1 24`
#python3 ../python/batchsize.py "../out/batchsize-$suffix.csv" $count "../out/batchsize-$suffix.png"
#python3 ../python/compare_batchsize.py "../out/batchsize-avx256.csv" "../out/batchsize-$suffix.csv" $count "../out/batchsize-compare.png"
