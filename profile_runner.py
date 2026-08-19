#!/usr/bin/env python3

import argparse
import subprocess
import psutil
import time
import os
import csv

parser = argparse.ArgumentParser()

parser.add_argument("--method", required=True)
parser.add_argument("--env", required=True)
parser.add_argument("--exp", required=True)
parser.add_argument("--id", required=True)
parser.add_argument("--command", required=True)

args = parser.parse_args()

os.makedirs("cpu_mem_profile", exist_ok=True)

outfile = (
    f"cpu_mem_profile/"
    f"{args.method}_{args.env}{args.id}_{args.exp}.csv"
)

###############################################################
# Launch experiment
###############################################################

import shlex

command = shlex.split(args.command)

process = subprocess.Popen(command)

p = psutil.Process(process.pid)

# initialize cpu counter
p.cpu_percent(interval=None)

start = time.time()

peak_rss = 0

with open(outfile, "w", newline="") as f:

    writer = csv.writer(f, delimiter=';')
    writer.writerow([
        "time_s",
        "cpu_percent",
        "rss_mb",
        "vms_mb",
        "num_threads"
    ])

    while process.poll() is None:

        try:

            cpu = p.cpu_percent(interval=0.1)

            mem = p.memory_info()

            rss = mem.rss / 1024**2
            vms = mem.vms / 1024**2

            peak_rss = max(peak_rss, rss)

            writer.writerow([
                round(time.time()-start,4),
                round(cpu,2),
                round(rss,2),
                round(vms,2),
                p.num_threads()
            ])

        except psutil.NoSuchProcess:
            break

exit(process.returncode)