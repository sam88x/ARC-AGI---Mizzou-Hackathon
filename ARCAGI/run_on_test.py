"""
Run trained model on test set or custom JSON files.

This script allows you to:
1. Run on the official ARC-AGI test set (100 tasks)
2. Run on the evaluation set (400 tasks)  
3. Run on a custom JSON file with new puzzles

The model trains from scratch for each task (no pre-training),
so this works the same way as training - just on different data.
"""

import os
import sys
import time
import json
import argparse
import multiprocessing

import torch

import solve_task

def run_on_dataset(split, output_file, n_iterations=2000, verbose=True):
    """
    Run the model on a specific dataset split.
    
    Args:
        split (str): 'training', 'evaluation', or 'test'
        output_file (str): Where to save the results
        n_iterations (int): Number of training iterations per task
        verbose (bool): Whether to print progress
    """
    
    # Setup
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()
    
    print(f"\n{'='*70}")
    print(f"Running on {split.upper()} set")
    print(f"{'='*70}")
    print(f"GPUs available: {n_gpus}")
    print(f"CPUs available: {n_cpus}")
    print(f"Training iterations per task: {n_iterations}")
    print(f"Output file: {output_file}")
    print(f"{'='*70}\n")
    
    # Load the dataset
    challenges_file = f'dataset/arc-agi_{split}_challenges.json'
    
    if not os.path.exists(challenges_file):
        print(f"❌ ERROR: File not found: {challenges_file}")
        print(f"Make sure you have the ARC-AGI dataset in the dataset/ directory")
        return
    
    with open(challenges_file, 'r') as f:
        problems = json.load(f)
    
    task_names = list(problems.keys())
    n_tasks = len(task_names)
    
    print(f"Found {n_tasks} tasks in {split} set")
    print(f"Starting parallel processing...\n")
    
    # Measure memory usage for each task
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]
    
    print("Phase 1: Measuring memory usage per task (2 iterations)...")
    memory_dict, _, _ = parallelize_runs(
        gpu_task_quotas, task_usages, 2, 
        task_names, split, n_gpus, n_cpus, n_tasks,
        verbose=verbose
    )
    
    # Sort tasks by memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)
    
    print(f"\nPhase 2: Running full training ({n_iterations} iterations)...")
    safe_gpu_memory_quotas = [memory_quota - 4 * 1024**3 for memory_quota in gpu_memory_quotas]
    _, solutions_dict, time_taken = parallelize_runs(
        safe_gpu_memory_quotas, task_memory_usages, n_iterations,
        task_names, split, n_gpus, n_cpus, n_tasks,
        verbose=verbose
    )
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(solutions_dict, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"Tasks solved: {len(solutions_dict)}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


def run_on_custom_json(input_file, output_file, n_iterations=2000, verbose=True):
    """
    Run the model on a custom JSON file.
    
    The JSON file should follow ARC-AGI format:
    {
        "task_id_1": {
            "train": [
                {"input": [[...]], "output": [[...]]}
            ],
            "test": [
                {"input": [[...]]}
            ]
        },
        ...
    }
    
    Args:
        input_file (str): Path to custom JSON file
        output_file (str): Where to save results
        n_iterations (int): Number of training iterations per task
        verbose (bool): Whether to print progress
    """
    
    # Setup
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()
    
    print(f"\n{'='*70}")
    print(f"Running on CUSTOM dataset")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"GPUs available: {n_gpus}")
    print(f"CPUs available: {n_cpus}")
    print(f"Training iterations per task: {n_iterations}")
    print(f"{'='*70}\n")
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"❌ ERROR: File not found: {input_file}")
        return
    
    # Load and validate the custom dataset
    try:
        with open(input_file, 'r') as f:
            problems = json.load(f)
        
        # Validate format
        for task_id, task in problems.items():
            if 'train' not in task or 'test' not in task:
                print(f"❌ ERROR: Task {task_id} missing 'train' or 'test' key")
                print("Expected format: {task_id: {'train': [...], 'test': [...]}}")
                return
            
            for example in task['train']:
                if 'input' not in example or 'output' not in example:
                    print(f"❌ ERROR: Training example in {task_id} missing 'input' or 'output'")
                    return
            
            for example in task['test']:
                if 'input' not in example:
                    print(f"❌ ERROR: Test example in {task_id} missing 'input'")
                    return
        
        print(f"✅ Valid ARC-AGI format detected")
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON file: {e}")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # Create a temporary file in the dataset directory for processing
    temp_file = f'dataset/custom_temp_{int(time.time())}.json'
    with open(temp_file, 'w') as f:
        json.dump(problems, f)
    
    task_names = list(problems.keys())
    n_tasks = len(task_names)
    
    print(f"Found {n_tasks} tasks in custom dataset")
    print(f"Starting parallel processing...\n")
    
    # Measure memory usage
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]
    
    print("Phase 1: Measuring memory usage per task (2 iterations)...")
    
    # For custom files, we need to temporarily modify the solve_task to use custom path
    # Instead, we'll use a simpler approach: copy to standard location
    split = "custom_temp"
    
    memory_dict, _, _ = parallelize_runs(
        gpu_task_quotas, task_usages, 2,
        task_names, split, n_gpus, n_cpus, n_tasks,
        verbose=verbose,
        custom_file=temp_file
    )
    
    # Sort tasks by memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)
    
    print(f"\nPhase 2: Running full training ({n_iterations} iterations)...")
    safe_gpu_memory_quotas = [memory_quota - 4 * 1024**3 for memory_quota in gpu_memory_quotas]
    _, solutions_dict, time_taken = parallelize_runs(
        safe_gpu_memory_quotas, task_memory_usages, n_iterations,
        task_names, split, n_gpus, n_cpus, n_tasks,
        verbose=verbose,
        custom_file=temp_file
    )
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(solutions_dict, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"Tasks solved: {len(solutions_dict)}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


def parallelize_runs(gpu_quotas, task_usages, n_iterations, task_names, split, n_gpus, n_cpus, n_tasks, verbose=False, custom_file=None):
    """
    Run tasks in parallel across GPUs.
    (Simplified version of the function from parallel_train.py)
    """
    t = time.time()
    gpu_quotas = gpu_quotas[:]
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]

    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        # Job monitoring loop
        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())

            # If a job finishes, release its quota
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(f"{task_names[i]} finished on gpu {process_gpu_ids[i]}")
            
            # Start new jobs if there is quota
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        
                        # Use custom split name if provided
                        actual_split = split if not custom_file else "custom_temp"
                        args = (task_names[i], actual_split, 1e20, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(f"{task_names[i]} started on gpu {process_gpu_ids[i]}")
            time.sleep(1)

        if not error_queue.empty():
            raise ValueError(error_queue.get())

        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)

    time_taken = time.time() - t
    return memory_dict, solutions_dict, time_taken


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run ARC-AGI model on test set or custom JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on official test set (100 tasks)
  python run_on_test.py --split test --output test_submission.json
  
  # Run on evaluation set (400 tasks)
  python run_on_test.py --split evaluation --output eval_submission.json
  
  # Run on custom JSON file
  python run_on_test.py --custom my_puzzles.json --output my_results.json
  
  # Use fewer iterations for faster results
  python run_on_test.py --split test --output test_submission.json --iterations 1000
        """
    )
    
    parser.add_argument('--split', type=str, choices=['training', 'evaluation', 'test'],
                        help='Which ARC-AGI split to run on')
    parser.add_argument('--custom', type=str,
                        help='Path to custom JSON file with puzzles')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for results')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='Number of training iterations per task (default: 2000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.split and args.custom:
        print("❌ ERROR: Cannot specify both --split and --custom")
        print("Use --split for ARC-AGI datasets or --custom for your own JSON file")
        sys.exit(1)
    
    if not args.split and not args.custom:
        print("❌ ERROR: Must specify either --split or --custom")
        print("\nExamples:")
        print("  python run_on_test.py --split test --output test_results.json")
        print("  python run_on_test.py --custom my_puzzles.json --output my_results.json")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # Run on the specified dataset
    if args.split:
        run_on_dataset(args.split, args.output, args.iterations, verbose)
    else:
        run_on_custom_json(args.custom, args.output, args.iterations, verbose)
