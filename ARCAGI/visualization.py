import os

import matplotlib.pyplot as plt
import numpy as np
import torch


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


color_list = np.array([
    [0, 0, 0],  # black
    [30, 147, 255],  # blue
    [249, 60, 49],  # red
    [79, 204, 48],  # green
    [255, 220, 0],  # yellow
    [153, 153, 153],  # gray
    [229, 58, 163],  # magenta
    [255, 133, 27],  # orange
    [135, 216, 241],  # light blue
    [146, 18, 49],  # brown
])

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

def plot_problem(logger):
    """
    Draw a plot of an ARC-AGI problem, and save it in plots/
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """

    # Put all the grids beside one another on one grid
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    pixels = 255+np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],mode_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid
    pixels = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
    
    os.makedirs("plots/", exist_ok=True)

    # Plot the combined grid and make gray dividers between the grid cells, arrows, and a question mark for unsolved examples.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for example_num in range(n_examples):
        for mode_num, mode in enumerate(('input', 'output')):
            if example_num < n_train:
                subsplit = 'train'
                subsplit_example_num = example_num
            else:
                subsplit = 'test'
                subsplit_example_num = example_num - n_train
            ax.arrow((2*n_y+8)-3-0.5, (2*n_x+2)*example_num+1+n_x-0.5, 6, 0, width=0.5, fc='k', ec='k', length_includes_head=True)
            if subsplit == 'test' and mode == 'output':
                ax.text((2*n_y+8)+4+n_y-0.5, (2*n_x+2)*example_num+1+n_x-0.5, '?', size='xx-large', ha='center', va='center')
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            for xline in range(grid.shape[0]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]-0.5, (2*n_y+8)*mode_num+4+n_y+grid.shape[1]-0.5),
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(grid.shape[1]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]-0.5, (2*n_x+2)*example_num+1+n_x+grid.shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    plt.axis('off')
    plt.savefig('plots/' + logger.task.task_name + '_problem.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_solution(logger, fname=None):
    """
    Draw a plot of a model's solution to an ARC-AGI problem, and save it in plots/
    Draws four plots: A model output sample, the mean of samples, and the top two most common samples.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y

    # Four plotted solutions
    solutions_list = [
            torch.softmax(logger.current_logits, dim=1).cpu().numpy(),
            torch.softmax(logger.ema_logits, dim=1).cpu().numpy(),
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
            ]
    masks_list = [
            (logger.current_x_mask, logger.current_y_mask),
            (logger.ema_x_mask, logger.ema_y_mask),
            None,
            None,
            ]
    solutions_labels = [
            'sample',
            'sample average',
            'guess 1',
            'guess 2',
            ]
    n_plotted_solutions = len(solutions_list)

    # Put all the grids beside one another on one grid
    pixels = 255+np.zeros([n_test, 2*n_x+2, n_plotted_solutions, 2*n_y+8, 3], dtype=np.uint8)
    shapes = []
    for subsplit_example_num in range(n_test):
        subsplit = 'test'
        example_num = subsplit_example_num + n_train
        shapes.append([])

        for solution_num, (solution, masks, label) in enumerate(zip(solutions_list, masks_list, solutions_labels)):
            grid = np.array(solution[subsplit_example_num])  # c, x, y if 'sample' in label else x, y, c
            if 'sample' in label:
                grid = np.einsum('dxy,dc->xyc', grid, color_list[logger.task.colors])  # x, y, c
                if logger.task.in_out_same_size or logger.task.all_out_same_size:
                    x_length = logger.task.shapes[example_num][1][0]
                    y_length = logger.task.shapes[example_num][1][1]
                else:
                    x_length = None
                    y_length = None
                x_start, x_end = logger._best_slice_point(masks[0][subsplit_example_num,:], x_length)
                y_start, y_end = logger._best_slice_point(masks[1][subsplit_example_num,:], y_length)
                grid = grid[x_start:x_end,y_start:y_end,:]  # x, y, c
                grid = np.clip(grid, 0, 255).astype(np.uint8)
            else:
                grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
                grid = convert_color(grid)  # x, y, c

            shapes[subsplit_example_num].append((grid.shape[0], grid.shape[1]))
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[subsplit_example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],solution_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid

    pixels = pixels.reshape([n_test*(2*n_x+2), n_plotted_solutions*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, and labels.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for subsplit_example_num in range(n_test):
        for solution_num in range(n_plotted_solutions):
            subsplit = 'test'
            grid = np.array(solutions_list[solution_num][subsplit_example_num])  # x, y
            shape = shapes[subsplit_example_num][solution_num]
            for xline in range(shape[0]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]-0.5, (2*n_y+8)*solution_num+4+n_y+shape[1]-0.5),
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(shape[1]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]-0.5, (2*n_x+2)*subsplit_example_num+1+n_x+shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    for solution_num, solution_label in enumerate(solutions_labels):
        ax.text((2*n_y+8)*solution_num+4+n_y-0.5, -3, solution_label, size='xx-small', ha='center', va='center')
    plt.axis('off')
    if fname is None:
        fname = 'plots/' + logger.task.task_name + '_solutions.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


