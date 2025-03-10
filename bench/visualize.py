import csv
import glob
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

makedirs("bench/vis", exist_ok=True)

for file in glob.glob("bench/ambit_*.csv"):
    fileKey = file[12:len(file)-4]

    benchmarkCols = [
        "t_opt", "t_runner", "t_extractor", "t_compiler",
        "mig_size", "mig_opt_size", "mig_pis", "mig_pos",
        "instruction_count",
        "egraph_classes", "egraph_nodes", "egraph_size"
    ]

    benchmarks = []
    rows = []
    for row in csv.reader(open(file), delimiter="\t"):
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
                benchmarkResult[benchmarkCols[colIdx]] = int(row[i])
                colIdx += 1
                i += 1

        rowData["results"] = benchmarkResults
        rows.append(rowData)


    for col in benchmarkCols:
        groups = benchmarks
        values = {}

        # Plot data
        width = 1 / (len(rows) + 1)
        x = np.arange(len(benchmarks))

        fig, ax = plt.subplots(layout="constrained")
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

        plt.legend()
        plt.savefig("bench/vis/" + fileKey + "_" + col + ".png", bbox_inches="tight")
        plt.close()
