#!/bin/sh

OUT_FILE="ambit_uprograms.csv"

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
  'div4'
  # div: unsigned restoring divider (quotient = a / b), added in utils.h.
  #   div4: ~41s with rewriting (499 instr).
  #   div8: only feasible with rewriting DISABLED -> run with AMBIT_REWRITE=0
  #         (~84s, 1543 unoptimized instr). With rewriting it does not finish.
  #   div16/div32: infeasible (extraction ~O(n^3.5); div16 ~hours even with
  #         rewriting disabled). Left empty.
  # 'eq4' 'eq8' 'eq16' 'eq32' \
  # 'ge4' 'ge8' 'ge16' 'ge32' \
  # 'sub4' 'sub8' 'sub16' 'sub32'
  # Recompiled due to PI order change in utils.h (a first, then b)
  # Already completed: ifelse4-64 mul4-32 min4-32 max4-32 add4-32 gt4-32 fa abs4-32 bitcount4-32
  # 'ifelse64' 'mul64' 'sub64' 'eq64' 'ge64' 'min64' 'max64' 'add64' 'gt64' 'abs64' 'bitcount64'
  # 'ntk/ctrl.aig' 'ntk/dec.aig' 'ntk/int2float.aig' 'ntk/router.aig'
do
  benchmarkName=$(basename "$benchmark" | cut -d. -f1)
  echo "Running $benchmarkName..."
  out=$(./build/lime_ambit_benchmark "$benchmark")
  if [ $? -eq 0 ]; then
    printf "\t\t%s\t%s" "$benchmarkName" "$out" >> "$OUT_FILE"
  else
    echo "failed (output: $out)"
  fi
done

echo >> "$OUT_FILE"

echo "Generating visualizations..."
python3 visualize.py
