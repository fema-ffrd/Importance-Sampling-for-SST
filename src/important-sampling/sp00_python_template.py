# -*- coding: utf-8 -*-


# Script Description ##########################################################
"""
A function-based python script template for tool development in MBI's Boathouse

"""

# Imports #####################################################################

# standard packages (remove packages when not required)
import multiprocessing

# Functions ###################################################################


def example_function(low: int, high: int) -> list:
    """example function that creates a list of numbers

    Args
        low (int): lower end of range
        high (int): higher end of range

    Returns
        example_list (list): list of numbers
    """

    # perform logic
    example_list = list(range(low, high))

    return example_list


def wrapper_example_function(global_low: int, global_high: int, increment_val: int) -> None:
    """an example of how to excecute a function in parallel"""

    # created grouped parameters
    grouped_parameters = []

    current_low = global_low
    while current_low < global_high:
        current_high = min(current_low + increment_val, global_high)
        grouped_parameters.append((current_low, current_high))
        current_low = current_high

    # make parallel call
    # num_cores = multiprocessing.cpu_count() -1 # for using all but one of the cores
    num_cores = 2 # for using a specific number of cores
    print(f"Setting up parallel processing with {num_cores} cores")

    with multiprocessing.Pool(processes=num_cores) as pool:
        results_list = pool.starmap(example_function, grouped_parameters)

        pool.close()
        pool.join()

    # splat results list
    results_list = [item for sublist in results_list for item in sublist]

    return results_list
