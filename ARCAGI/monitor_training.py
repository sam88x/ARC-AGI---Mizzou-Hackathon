#!/usr/bin/env python3
"""
Real-time monitoring for parallel_train.py
Run this in a separate terminal while parallel_train.py is running.
"""

import time
import subprocess
import os
import json
from datetime import datetime, timedelta

def get_gpu_stats():
    """Get GPU utilization and memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, util, mem_used, mem_total, temp = line.split(', ')
                gpu_stats.append({
                    'index': int(idx),
                    'utilization': int(util),
                    'memory_used': float(mem_used),
                    'memory_total': float(mem_total),
                    'temperature': int(temp)
                })
        return gpu_stats
    except Exception as e:
        return None

def get_submission_progress():
    """Check how many tasks have been solved."""
    try:
        if os.path.exists('submission.json'):
            with open('submission.json', 'r') as f:
                submission = json.load(f)
            return len(submission)
        return 0
    except:
        return 0

def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))

def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')

def monitor_training(refresh_seconds=2):
    """Main monitoring loop."""
    start_time = time.time()
    total_tasks = 400  # ARC-AGI training set size
    
    print("Starting monitoring... Press Ctrl+C to stop\n")
    
    try:
        while True:
            clear_screen()
            
            elapsed = time.time() - start_time
            
            # Header
            print("="*80)
            print(f"ARC-AGI PARALLEL TRAINING MONITOR".center(80))
            print(f"Runtime: {format_time(elapsed)}".center(80))
            print("="*80)
            
            # GPU Stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print("\nGPU STATUS:")
                print("-"*80)
                print(f"{'GPU':<6} {'Utilization':<15} {'Memory':<25} {'Temp':<10}")
                print("-"*80)
                
                total_util = 0
                for gpu in gpu_stats:
                    util_bar = '█' * (gpu['utilization'] // 5) + '░' * (20 - gpu['utilization'] // 5)
                    mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
                    mem_str = f"{gpu['memory_used']/1024:.1f}/{gpu['memory_total']/1024:.1f} GB ({mem_pct:.1f}%)"
                    
                    util_color = '✓' if gpu['utilization'] > 80 else '⚠' if gpu['utilization'] > 50 else '✗'
                    
                    print(f"{gpu['index']:<6} {util_color} {gpu['utilization']:>3}% {util_bar:<20} {mem_str:<25} {gpu['temperature']:>3}°C")
                    total_util += gpu['utilization']
                
                avg_util = total_util / len(gpu_stats) if gpu_stats else 0
                print("-"*80)
                print(f"Average GPU Utilization: {avg_util:.1f}%")
                
                if avg_util < 50:
                    print("⚠️  WARNING: Low GPU utilization - training may not be using GPUs effectively")
                elif avg_util > 90:
                    print("✅ EXCELLENT: GPUs are fully utilized!")
                else:
                    print("✓  GOOD: GPUs are being used")
            else:
                print("\n⚠️  Cannot read GPU stats. Make sure nvidia-smi is available.")
            
            # Progress
            tasks_completed = get_submission_progress()
            progress_pct = (tasks_completed / total_tasks) * 100
            progress_bar = '█' * int(progress_pct // 2) + '░' * (50 - int(progress_pct // 2))
            
            print("\nTRAINING PROGRESS:")
            print("-"*80)
            print(f"Tasks Completed: {tasks_completed} / {total_tasks} ({progress_pct:.1f}%)")
            print(f"[{progress_bar}]")
            
            if tasks_completed > 0 and elapsed > 0:
                tasks_per_sec = tasks_completed / elapsed
                remaining_tasks = total_tasks - tasks_completed
                eta_seconds = remaining_tasks / tasks_per_sec if tasks_per_sec > 0 else 0
                print(f"Speed: {tasks_per_sec:.2f} tasks/sec")
                print(f"Estimated Time Remaining: {format_time(eta_seconds)}")
                print(f"Estimated Completion: {(datetime.now() + timedelta(seconds=eta_seconds)).strftime('%Y-%m-%d %H:%M:%S')}")
            
            print("-"*80)
            
            # Cost estimation (for g4dn.12xlarge at $3.912/hour)
            cost_per_hour = 3.912
            elapsed_hours = elapsed / 3600
            current_cost = elapsed_hours * cost_per_hour
            
            if tasks_completed > 0 and elapsed > 0:
                tasks_per_sec = tasks_completed / elapsed
                remaining_tasks = total_tasks - tasks_completed
                eta_seconds = remaining_tasks / tasks_per_sec if tasks_per_sec > 0 else 0
                total_hours = (elapsed + eta_seconds) / 3600
                estimated_total_cost = total_hours * cost_per_hour
                
                print(f"\nCOST ESTIMATE (g4dn.12xlarge @ ${cost_per_hour}/hour):")
                print("-"*80)
                print(f"Current Cost: ${current_cost:.2f}")
                print(f"Estimated Total Cost: ${estimated_total_cost:.2f}")
                print(f"Budget Remaining: ${100 - estimated_total_cost:.2f} / $100")
            else:
                print(f"\nCurrent Cost: ${current_cost:.2f} / $100")
            
            print("="*80)
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refreshing every {refresh_seconds}s")
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Total runtime: {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    import sys
    refresh = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    monitor_training(refresh)
