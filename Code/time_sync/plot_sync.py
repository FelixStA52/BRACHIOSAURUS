import csv
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_relative_difference(box_number):
    filename = f"time{box_number}.csv"
    if not os.path.exists(filename):
        print(f"Data file for box {box_number} not found.")
        return
    
    iterations = []
    clock_times = []
    real_times = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            iterations.append(int(row[0]))
            clock_times.append(int(row[1]))
            real_times.append(int(row[2]))
    
    total_diff = [c - r for c, r in zip(clock_times, real_times)]
    relative_diff = [np.nan]
    for i in range(1, len(total_diff)):
        relative_diff.append(total_diff[i] - total_diff[i-1])
    
    cut_index = len(iterations)
    for i in range(1, len(iterations)):
        if iterations[i] == 0:
            cut_index = i
            break
    
    valid_x = iterations[1:cut_index]
    valid_y = relative_diff[1:cut_index]
    
    if not valid_x:
        print("No valid data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(valid_x, valid_y, 'b-', marker='o', markersize=4)
    plt.title(f'Clock Synchronization Differences for Box {box_number}\n'
             f'[First {len(valid_x)} measurements before reboot]')
    plt.xlabel('Iteration Number')
    plt.ylabel('Relative Time Difference (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()