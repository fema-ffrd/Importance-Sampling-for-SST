# -*- coding: utf-8 -*-

# Script Description ##########################################################
"""
A python template for running functions defined in the src directory and storing
project specific parameters in version control. Note logic and execution of the 
logic is now separated. This allows for project specific parameters to be tracked
and task specific logic to be integrated with Boathouse Tools.

"""

# Imports #####################################################################

# standard packages (remove packages when not required)
from datetime import datetime
import os

from sp00_python_template import example_function

# Functions ###################################################################


if __name__ == "__main__":

    time1 = datetime.now()
    print(f"starting, {time1}")

    # define relative paths to the project's root directory
    currDir = os.path.dirname(os.path.realpath(__file__))
    rootDir = os.path.abspath(os.path.join(currDir, ".."))

    my_file = os.path.join(rootDir, "data", "example_data.csv")

    # set project/testing specific arguments
    LOW = 0
    HIGH = 10

    # call imported functions
    results_list = example_function(LOW, HIGH)

    print(f"results_list from python runner template: {results_list}")

    time2 = datetime.now()
    print(f"complete, {time2}")
    print(f"elapsed, {time2-time1}")