"""Sample total RSS of a process tree every 2 s (issue #646 audit)."""

import os
import sys
import time

import psutil

root = psutil.Process(int(sys.argv[1]))
out = open(sys.argv[2], "w", buffering=1)
out.write("t,total_rss,n_procs,max_proc_rss,load1\n")
t0 = time.time()
while root.is_running():
    total = mx = n = 0
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        break
    for p in procs:
        try:
            r = p.memory_info().rss
        except psutil.Error:
            continue
        total += r
        mx = max(mx, r)
        n += 1
    out.write(f"{time.time() - t0:.1f},{total},{n},{mx},{os.getloadavg()[0]:.1f}\n")
    time.sleep(2)
