#!/bin/sh

OUT_FILE="ambit.csv"

mkdir -p bench
cd bench || exit

# Read parameters
printf "Description: "
read -r description
time=$(date "+%Y-%m-%d %H:%M:%S")
rev=$(git rev-parse --short HEAD)

# Perform build
mkdir -p build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release ../..
make lime_ambit_benchmark || exit
cd ..

printf "%s\t%s\t%s" "$time" "$rev" "$description" >> "$OUT_FILE"

for benchmark in \
  'mux' 'fa' 'add2' 'add3' 'add4' 'add32' 'add64' 'gt' 'mul2' 'mul6' 'pop2' 'pop4' 'kogge_stone' \
  'ntk/ctrl.aig' 'ntk/dec.aig' 'ntk/int2float.aig' 'ntk/router.aig'
do
  benchmarkName=$(basename "$benchmark" | cut -d. -f1)
  echo "Running $benchmarkName..."
  out=$(./build/lime_ambit_benchmark "$benchmark")
  if [ $? -eq 0 ]; then
  printf "\t\t%s\t%s" "$benchmarkName" "$out" >> "$OUT_FILE"
    fi
done

echo >> "$OUT_FILE"

echo "Generating visualizations..."
python3 visualize.py
