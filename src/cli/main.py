#!/usr/bin/env python3

# Script Description ##########################################################
"""
A command line interface for important-sampling.

To test your command line tools during development try the commands below. 
You may need to comment out the authorizor import in the __init__.py file in the 
important-sampling directory if a container doesn't exist 
in one of the ECR repositories identified in the github workflows yml files.

    $ python src/important-sampling_cli.py --help
    $ python src/important-sampling_cli.py example-function --help
"""

import click
from important-sampling.sp00_python_template import example_function as example_function_impl
from important-sampling.sp00_python_template import wrapper_example_function as wrapper_example_function_impl


@click.group()
def main():
    """The main entry point for the command line interface."""


@main.command()
@click.option(
    "--low",
    help="lower end of range",
)
@click.option(
    "--high",
    help="higher end of range",
)
def example_function(low: int, high: int) -> None:
    """example function that creates a list of numbers

    Args
        low (int): lower end of range
        high (int): higher end of range

    Returns
        None
    """
    try:
        example_function_impl(low, high)
    except Exception as exc:
        print(f"example_function failed with the following exception: {str(exc)}")
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies


@main.command()
@click.option(
    "--global_low",
    help="lower end of range",
)
@click.option(
    "--global_high",
    help="higher end of range",
)
@click.option(
    "--increment_val",
    help="increment value for multiprocessing",
)
def wrapper_example_function(global_low: int, global_high: int, increment_val: int) -> None:
    """multiprocessing wrapper for example function that creates a list of numbers

    Args
        global_low (int): lower end of range
        global_high (int): higher end of range
        increment_val (int): increment value for multiprocessing

    Returns
        None
    """
    try:
        wrapper_example_function_impl(global_low, global_high, increment_val)
    except Exception as exc:
        print(f"wrapper_example_function failed with the following exception: {str(exc)}")
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies


# Script ######################################################################

if __name__ == "__main__":
    # run the function
    # pylint does not understand click decorators
    # pylint: disable=no-value-for-parameter
    main()
