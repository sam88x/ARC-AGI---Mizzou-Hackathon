import json
import bisect

import torch
import numpy as np
import matplotlib.pyplot as plt

import preprocessing


"""
This code computes the pass@n accuracy over all the tasks over time.
"""


class ValueSortedDict:
    """
    A value-sorted dict that supports:
     - insertion/removal
     - indexing by key
     - indexing by rank of value
     - figuring out the rank of a key
    """

    def __init__(self):
        self.sorted_list = []
        self.key_to_value = {}

    def insert(self, key, value):
        """
        Insert a key, value pair into the data structure.
        """
        if key in self.key_to_value:
            self.remove(key)
        
        # Use bisect to maintain order by value
        bisect.insort(self.sorted_list, (value, key))
        self.key_to_value[key] = value

    def get(self, key, default=0):
        """
        Index by key.
        """
        if key in self.key_to_value:
            return self.key_to_value.get(key)
        return default

    def remove(self, key):
        """
        Remove by key.
        """
        if key in self.key_to_value:
            value = self.key_to_value.pop(key)
            index = bisect.bisect_left(self.sorted_list, (value, key))
            if index < len(self.sorted_list) and self.sorted_list[index] == (value, key):
                self.sorted_list.pop(index)

    def items(self):
        """
        Get the key/value pairs sorted by value.
        """
        return [(key, value) for value, key in self.sorted_list]

    def get_by_index(self, index):
        """
        Get the key/value pair for the nth ranked value.
        """
        if -len(self.sorted_list) <= index < len(self.sorted_list):
            value, key = self.sorted_list[index]
            return key, value
        raise IndexError("Index out of range")

    def find_key(self, key):
        """
        Figure out the value rank from a given key.
        """
        if key not in self.key_to_value:
            return -1
        sorted_keys = [key for value, key in self.sorted_list]
        return sorted_keys.index(key)

def get_accuracy(true_solution_hashes, fname='predictions.npz'):
    """
    Compute the pass@n accuracy over time, of a training run whose
    solution history is stored in a file outputted by train.py.
    Args:
        true_solution_hashes (list[int]): The hashes of the ground truth solutions.
        fname (str): The train.py output file that has the solution history.
    Returns:
        Tensor: A (iteration, pass@n) tensor that gives the accuracy at any iteration
                for any number of attempts.
    """

    stored_data = np.load(fname, allow_pickle=True)
    solution_contribution_logs = stored_data['solution_contribution_logs']
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_contribution_logs)
    n_iterations = len(solution_contribution_logs[0])
    n_attempts = n_iterations

    print("Plotting accuracy for " + str(n_tasks) + " tasks.")

    # First mark the number of attempts required to solve a task at every iteration.
    # Accumulate markings for all tasks.
    pass_at_n = np.zeros([n_iterations, n_attempts])
    for task_num in range(n_tasks):
        true_hash = true_solution_hashes[task_num] >> 16
        solution_scores = ValueSortedDict()
        for iteration_num in range(n_iterations):
            for i in range(2):
                hashed, score = solution_contribution_logs[task_num][iteration_num][i]
                hashed = int(hashed) >> 16
                original_score = torch.tensor(solution_scores.get(hashed, default=-10000))
                score = torch.tensor(score)
                new_score = float(torch.logaddexp(score, original_score))
                solution_scores.insert(hashed, new_score)
            solution_index = solution_scores.find_key(true_hash)
            if solution_index != -1:
                solution_index = len(solution_scores.sorted_list)-1-solution_index
                if solution_index < n_attempts:
                    pass_at_n[iteration_num,solution_index] += 1

    # If solved at attempt n, then also solved at attempt m for all m>n.
    pass_at_n = np.cumsum(pass_at_n, axis=1)  # iteration, @n

    # Divide by number of tasks
    pass_at_n = pass_at_n / n_tasks
    return pass_at_n

def plot_accuracy(pass_at_n):
    """
    Plot the accuracy curve, for pass@n for n from {1, 2, 5, 10, 100, 1000, 2000}.
    Args:
        pass_at_n (Tensor): A (iteration, pass@n) tensor that gives the accuracy
                at any iteration for any number of attempts.
    """
    n_iterations, n_attempts = pass_at_n.shape

    fig, ax = plt.subplots()

    ax.plot(np.arange(n_iterations), pass_at_n[:,0], 'r-', label='pass@1')
    ax.plot(np.arange(n_iterations), pass_at_n[:,1], 'k-', label='pass@2')
    ax.plot(np.arange(n_iterations), pass_at_n[:,4], 'g-', label='pass@5')
    ax.plot(np.arange(n_iterations), pass_at_n[:,9], 'b-', label='pass@10')
    ax.plot(np.arange(n_iterations), pass_at_n[:,99], 'c-', label='pass@100')
    ax.plot(np.arange(n_iterations), pass_at_n[:,999], 'm-', label='pass@1000')
    # ax.plot(np.arange(n_iterations), pass_at_n[:,1999], 'y-', label='pass@2000')

    ax.legend()
    
    plt.xlabel("step")
    plt.ylabel("accuracy")
    ax.grid(which='both', color='0.65', linewidth=0.8, linestyle='-')
    
    plt.savefig('accuracy_curve_at_n.png', bbox_inches='tight')
    plt.close()

def print_accuracy(pass_at_n):
    """
    Print the accuracy@n for various training iterations for various values of n.
    Args:
        pass_at_n (Tensor): A (iteration, pass@n) tensor that gives the accuracy
                at any iteration for any number of attempts.
    """
    n_iterations, n_attempts = pass_at_n.shape
    for iter_num in [100, 200, 300, 400, 500, 750, 1000]:#, 1250, 1500, 2000]:
        for attempt_num in [1, 2, 5, 10, 100, 1000]:
            print('iteration ' + str(iter_num) + ', ' + str(attempt_num) + ' attempts: accuracy = ' + str(float(pass_at_n[iter_num-1, attempt_num-1])))

if __name__ == "__main__":
    task_nums = list(range(400))
    split = input('Enter which split you want to find the task in (training, evaluation, test): ')
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    true_solution_hashes = [task.solution_hash for task in tasks]
    pass_at_n = get_accuracy(true_solution_hashes, fname='predictions.npz')
    plot_accuracy(pass_at_n)
    print_accuracy(pass_at_n)
