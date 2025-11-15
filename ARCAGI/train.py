import time

import numpy as np
import torch

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import multiprocessing as mp


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_device('cpu')
# torch.set_default_device('cuda')


def train_single_task(args):
    """
    Train one ARC task end-to-end in a separate process.
    Returns (task_index, solution_hash, maybe other info).
    """
    split, task_num, n_iterations = args

    print(f"{task_num=} Starting training...")

    # Preprocess just this task
    tasks = preprocessing.preprocess_tasks(split, [task_num])
    task = tasks[0]

    # Build model / optimizer / logger just for this task
    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    train_history_logger = solution_selection.Logger(task)

    # Optional: skip plotting inside workers if it’s heavy
    # visualization.plot_problem(train_history_logger)

    for train_step in range(n_iterations):
        take_step(task, model, optimizer, train_step, train_history_logger)

    # Optional: also skip plotting here if you only care about final solutions
    # visualization.plot_solution(train_history_logger)

    # Save predictions for this single task (into a per-task file or shared location)
    # You can either:
    #   - write a per-task prediction file here, or
    #   - return the logger and aggregate later.
    # Here I’ll just return the info needed to build the global submission.
    print(f"{task_num=} Finished training...")
    return {
        "task_num": task_num,
        "solution_hash": task.solution_hash,
        "logger": train_history_logger,
    }

def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    """
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def take_step(task, model, optimizer, train_step, train_history_logger):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """

    optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

    # Compute the total KL loss
    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)

    # Compute the reconstruction error
    reconstruction_error = 0
    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            # Determine whether the grid size is already known.
            # If not, there is an extra term in the reconstruction error, corresponding to
            # the probability of reconstructing the correct grid size.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                # coefficient = 0.01**max(0, 1-train_step/100)
                coefficient = 0.2 + 0.8 * (1 - np.exp(-train_step / 200))
            else:
                coefficient = 1
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                y_log_partitions = []
                for length in range(1, x_mask.shape[1]+1):
                    x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                for length in range(1, y_mask.shape[1]+1):
                    y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            # Given that we have the correct grid size, get the reconstruction error of getting the colors right
            logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition  # given the correct grid size,
                    logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                    target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  # calculate the error for the colors.
                    logprobs[x_offset].append(logprob)
            logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
            # if grid_size_uncertain:
            #     coefficient = 0.1**max(0, 1-train_step/100)
            # else:
            #     coefficient = 1
            temperature = 1.5 if grid_size_uncertain else 1.0
            # logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient  # Aggregate for all possible grid sizes
            logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1)) * temperature
            reconstruction_error = reconstruction_error - logprob

    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)


if __name__ == "__main__":
    start_time = time.time()

    split = "test"   # "training", "evaluation", or "test"
    task_nums = list(range(120))
    n_iterations = 2000

    # How many processes you want to run in parallel.
    # On 16 vCPUs, 4–8 is usually a good starting point.
    n_procs = 16

    # IMPORTANT on some platforms
    mp.set_start_method("spawn", force=True)

    # Build argument list for each task
    args_list = [(split, task_num, n_iterations) for task_num in task_nums]

    with mp.Pool(processes=n_procs) as pool:
        results = pool.map(train_single_task, args_list)

    # Reconstruct the list in the original task order
    results.sort(key=lambda r: r["task_num"])

    # Extract loggers / solution hashes
    train_history_loggers = [r["logger"] for r in results]
    true_solution_hashes = [r["solution_hash"] for r in results]

    # Now that all tasks are trained, you can do the plotting & accuracy
    for logger in train_history_loggers:
        visualization.plot_solution(logger)

    solution_selection.save_predictions(train_history_loggers)
    solution_selection.plot_accuracy(true_solution_hashes)

    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))
    print("Time elapsed in seconds:", time.time() - start_time)
