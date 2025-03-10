#!/bin/sh

SMALL_OUT_FILE="bench/ambit_small.csv"
LARGE_OUT_FILE="bench/ambit_large.csv"

mkdir -p bench

# Read parameters
printf "Description: "
read -r description
time=$(date "+%Y-%m-%d %H:%M:%S")
rev=$(git rev-parse --short HEAD)

# Perform build
mkdir -p build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release ..
make lime_ambit_benchmark
cd ..

printf "%s\t%s\t%s" "$time" "$rev" "$description" >> $SMALL_OUT_FILE
printf "%s\t%s\t%s" "$time" "$rev" "$description" >> $LARGE_OUT_FILE

runBenchmark() {
  benchmark=$1
  outFile=$2

  benchmarkName=$(basename "$benchmark" | cut -d. -f1)
  echo "Running $benchmarkName..."
  out=$(./build/lime_ambit_benchmark "$benchmark")
  printf "\t\t%s\t%s" "$benchmarkName" "$out" >> "$outFile"
}


for benchmark in \
  'fa' 'add2' 'add3' 'add4' 'mul2' 'pop2' 'pop4'
do
  runBenchmark $benchmark $SMALL_OUT_FILE
done

for benchmark in \
  'add32' 'add64' 'mul6' 'bench/ntk/ctrl.aig' 'bench/ntk/dec.aig' 'bench/ntk/int2float.aig' 'bench/ntk/router.aig'
do
  runBenchmark $benchmark $LARGE_OUT_FILE
done

echo >> $SMALL_OUT_FILE
echo >> $LARGE_OUT_FILE

echo "Generating visualizations..."
python3 bench/visualize.py
