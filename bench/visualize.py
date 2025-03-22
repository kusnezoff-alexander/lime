import csv
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

makedirs("vis", exist_ok=True)

benchmarkMetrics = [
    "t_opt", "t_runner", "t_extractor", "t_compiler",
    "mig_size", "mig_opt_size", "mig_pis", "mig_pos",
    "instruction_count",
    "egraph_classes", "egraph_nodes", "egraph_size"
]
groupingMetric = "instruction_count"
groups = [30, 100]

# Read CSV file
benchmarks = []
rows = []
for row in csv.reader(open("ambit.csv"), delimiter="\t"):
    rowData = {
        "time": row[0],
        "commit": row[1],
        "description": row[2],
    }

    benchmarkResults = {}
    benchmarkResult = {}
    colIdx = 0

    i = 3
    benchmarkName = ""
    while i <= len(row):
        if i == len(row) or row[i] == "":
            oldBenchmarkName = benchmarkName
            if i != len(row):
                benchmarkName = row[i+1]
            if oldBenchmarkName != "":
                if oldBenchmarkName not in benchmarks:
                    benchmarks.append(oldBenchmarkName)
                benchmarkResults[oldBenchmarkName] = benchmarkResult
                benchmarkResult = {}

            colIdx = 0
            i += 2
        else:
            benchmarkResult[benchmarkMetrics[colIdx]] = float(row[i])
            colIdx += 1
            i += 1

    rowData["results"] = benchmarkResults
    rows.append(rowData)

# `benchmarks` is list of benchmark names
# `rows` is list of dicts of the form
# {
#   "time": ....
#   "commit": ...
#   "description": ...
#   "results": {
#     <benchmarkName>: {
#       <metric>: <value>
#       ...
#     }, ...
#   }
# }

# Now find maximum group metric value for each benchmark
maxMetrics = {}
for metric in benchmarkMetrics:
    metricValues = {}
    maxMetrics[metric] = metricValues
    for bench in benchmarks:
        m = 0
        for row in rows:
            m = max(row["results"][bench][metric], m)
        metricValues[bench] = m

# Determine the group for each benchmark
groupBenchmarks = [[] for i in range(len(groups) + 1)]
maxGroupMetrics = maxMetrics[groupingMetric]
for bench in benchmarks:
    found = False
    for i, upperLimit in enumerate(groups):
        if maxGroupMetrics[bench] < upperLimit:
            groupBenchmarks[i].append(bench)
            found = True
            break
    if not found:
        groupBenchmarks[len(groups)].append(bench)

# Then group the benchmarks
grouped = [[] for i in range(len(groups) + 1)]
for row in rows:
    groupedRows = []
    # Create new row entry in each grouped-entry
    for groupIdx, groupedEntry in enumerate(grouped):
        groupedRow = {k: row[k] for k in row.keys() - {"results"}}
        groupedRow["results"] = {}
        groupedRows.append(groupedRow)
        for bench in groupBenchmarks[groupIdx]:
            groupedRow["results"][bench] = row["results"][bench]
        groupedEntry.append(groupedRow)

def visualize(rows, file_prefix, benchmarks):
    for col in benchmarkMetrics:
        # Plot data
        width = 1 / (len(rows) + 1)
        x = np.arange(len(benchmarks))

        fig, ax = plt.subplots(layout="constrained")
        fig.set_size_inches(len(benchmarks), 5)
        ax.set_xticks(x + len(rows) * width / 2, benchmarks)
        ax.set_title(col)

        multiplier = 0
        for row in rows:
            results = row["results"]
            values = []
            for bench in benchmarks:
                if bench in results:
                    values.append(results[bench][col])
                else:
                    values.append(0)
            ax.bar(x + width * multiplier, values, width, label=row["description"], align="edge")
            multiplier += 1

        ax.axhline(color='black', linewidth=0.5)
        plt.legend()
        plt.savefig("vis/" + file_prefix + "_" + col + ".png", bbox_inches="tight")
        plt.close()

for i, groupedRows in enumerate(grouped):
    visualize(groupedRows, str(i), groupBenchmarks[i])

# Now normalize rows
for row in rows[1:]:
    for benchmark, metrics in row["results"].items():
        for metric, value in metrics.items():
            baseline = rows[0]["results"][benchmark][metric]
            metrics[metric] = 0 if value == 0 or baseline == 0 else (baseline - value) / baseline
visualize(rows[1:], "rel_diff", benchmarks)
