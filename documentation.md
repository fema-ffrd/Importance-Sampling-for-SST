# Project Documentation: Importance Sampling for SST

This documentation provides a detailed, semi-technical overview of each module and function in the project. It explains how the components are connected, what each function does, and the expected inputs and outputs. The goal is to help new developers and analysts understand the workflow and logic without requiring deep technical expertise.

---

## Table of Contents
- [Overview](#overview)
- [Module Summaries](#module-summaries)
  - [src/cli/](#srccli)
  - [src/evaluation/](#srcevaluation)
  - [src/importance-sampling-watersheds-pb/](#srcimportance-sampling-watersheds-pb)
  - [src/met/](#srcmet)
  - [src/toy_problem/](#srctoy_problem)
  - [src/transpose/](#srctranspose)
  - [src/utils/](#srcutils)
- [Function Details](#function-details)
- [Workflow Connections](#workflow-connections)

---

## Overview

The project is designed to improve the efficiency and accuracy of Stochastic Storm Transposition (SST) by using importance sampling techniques. The codebase is organized into modules that handle data preprocessing, sampling, statistical analysis, and visualization. Each module contains functions that perform specific tasks, and these functions are often used together in a sequence to process meteorological data and generate results.


---

## Module Summaries

### src/cli/
**Purpose:**
- Provides command-line tools to run key workflows in the project.
- Allows users to preprocess data, run sampling, and generate outputs without writing code.

**Key Script:**
- `main.py`: Entry point for CLI commands. Connects user input to the appropriate functions in other modules.

### src/evaluation/
**Purpose:**
- Contains tools for evaluating results and generating plots.
- Helps users visualize and interpret the output of the sampling and analysis workflows.

**Key Scripts:**
- `plots.py`: Functions for creating visualizations of results.
- `precipfrequency.py`, `simstats.py`: Statistical analysis and evaluation utilities.

### src/importance-sampling-watersheds-pb/
**Purpose:**
- Core logic for importance sampling and storm catalogue processing.
- Implements the main algorithms for sampling and statistical analysis.

**Key Scripts:**
- `main_preprocess_storm_catalogue.py`: Prepares storm data for analysis.
- `main_sampling.py`: Runs the importance sampling process.
- `modules/`: Contains specialized functions for computing statistics, processing data, and handling distributions.

### src/met/
**Purpose:**
- Handles meteorological data processing and event generation.
- Converts raw weather data into formats suitable for analysis.

**Key Scripts:**
- `aorc_event_generator.py`: Generates storm events from gridded precipitation data.
- `process_event_catalog.py`: Processes and summarizes event catalogues.

### src/toy_problem/
**Purpose:**
- Provides simplified models and test cases for development and demonstration.
- Useful for understanding the methodology and testing new ideas.

**Key Scripts:**
- `toy_testing.py`, `toy_testing runner.py`: Example workflows and test cases.
- `archive/`: Older or experimental scripts.

### src/transpose/
**Purpose:**
- Handles the transposition of storm data and sampling of storm centers.
- Provides utilities for shifting and sampling storm events in space.

**Key Scripts:**
- `transposer.py`: Functions for moving storm data.
- `sampling.py`: Methods for sampling storm centers.
- `precipdepths.py`: Computes precipitation depths for sampled storms.

### src/utils/
**Purpose:**
- Utility functions for geospatial operations, coordinate systems, and statistics.
- Used by multiple modules to handle common tasks.

**Key Scripts:**
- `zonalstats.py`, `crsutils.py`, `metutils.py`, `transposeutils.py`: Helper functions for spatial and statistical operations.

---

## Function Details

### src/cli/main.py

**Purpose:**
Provides a command-line interface (CLI) for running key project functions without writing code.

**How it works:**
- Uses the `click` library to define CLI commands.
- Each command (e.g., `example_function`, `wrapper_example_function`) is linked to a function in the codebase.
- When a user runs a command, the CLI parses the input options and calls the corresponding function, handling errors gracefully.

**Inputs/Outputs:**
- Inputs: Command-line options (e.g., `--low`, `--high`).
- Outputs: Runs the function and prints results or errors to the terminal.

---

### src/importance-sampling-watersheds-pb/main_preprocess_storm_catalogue.py

**Purpose:**
Prepares raw storm data for analysis by creating a storm catalogue.

**How it works:**
- Sets the working directory and input/output folders (user must update these paths).
- Calls `preprocess_storm_catalogue`, which reads NetCDF files containing precipitation data and processes them into a standardized catalogue format.
- The output is a folder containing the processed storm catalogue, ready for sampling.

**Inputs/Outputs:**
- Inputs: Folder with NetCDF storm files, variable name for precipitation data, output folder name.
- Outputs: A processed storm catalogue saved to disk.

---

### src/importance-sampling-watersheds-pb/main_sampling.py

**Purpose:**
Runs the main importance sampling workflow to generate and analyze storm samples.

**How it works (step-by-step):**
1. **Setup:**
   - Sets the working directory and paths to input data (storm catalogue, watershed, and domain files).
2. **Read Data:**
   - Loads the storm catalogue and geospatial files for the watershed and domain.
3. **Coordinate Matching:**
   - Ensures all spatial data uses the same coordinate system as the precipitation data.
4. **Extract Stats:**
   - Calculates statistics (e.g., bounds, centroids) for the watershed and domain polygons.
5. **Sampling:**
   - Sets up different statistical distributions for sampling storm centers (e.g., truncated normal, generalized normal, t-distribution, mixtures).
   - Runs large numbers of simulations to sample storm events using these distributions.
   - Saves the sampled storm data for later analysis.
6. **Depth Calculation:**
   - For each set of sampled storms, computes the precipitation depth over the watershed.
   - Saves the computed depths for later analysis.
7. **Evaluation:**
   - Reads the results and compares different sampling methods.
   - Generates plots showing the distribution of sampled points and frequency curves for precipitation depths.
   - Summarizes and visualizes the results to help users understand the impact of different sampling strategies.

**Inputs/Outputs:**
- Inputs: Processed storm catalogue, watershed/domain geospatial files, user-defined parameters for sampling.
- Outputs: Sampled storm data, computed precipitation depths, summary statistics, and plots.

---

## Workflow Connections

*This section will describe how the modules and functions work together in a typical workflow, from data preprocessing to final analysis and visualization. Flowcharts and diagrams will be added in future updates.*

---

*This document is a living resource and will be updated as the project evolves. For questions or suggestions, please contact the project maintainers.*
