# Importance Sampling for SST

## Project Overview

This project aims to improve Stochastic Storm Transposition (SST) through advanced importance sampling strategies, enhancing computational efficiency while maintaining statistical integrity. The repository provides tools and APIs for generating watershed-averaged precipitation-frequency curves using both simple and adaptive stratified importance sampling methods.

### Goals and Objectives
- Efficiently generate watershed-averaged precipitation-frequency curves
- Preserve statistical properties of precipitation data
- Optimize computational efficiency for large-scale hydrologic analysis
- Evaluate trade-offs between sampling complexity and computational demands

### Key Methods
- **Simple Method:** Uses a truncated bivariate normal distribution for importance sampling
- **Advanced Method:** Employs adaptive stratified importance sampling to create an optimal x-y importance grid

### Planned Activities
- Develop and test importance sampling methods
- Create adaptive grids for sampling
- Integrate approaches with current SST procedures
- Evaluate performance on specific watersheds

---

## Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/download)
- [Remote - Containers extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup Instructions
1. Ensure Docker Desktop is running.
2. Clone the repository:
   ```sh
   git clone https://github.com/fema-ffrd/Importance-Sampling-for-SST.git
   ```
3. Open the project folder in VSCode.
4. When prompted, click "Reopen in Container" to open the project inside the devcontainer. If not prompted, use the command palette (`Ctrl+Shift+P`) and select "Dev Containers: Rebuild and Reopen in Container".
5. Wait for the devcontainer to build and start (may take several minutes on first run).

### Adding Dependencies
Use the `env.yaml` file at the project root to manage top-level dependencies. Only specify major.minor versions to allow automatic patch updates. For complex dependencies, update the `.devcontainer/Dockerfile` and the production `Dockerfile` as needed.

---

## Project Organization

The repository contains the source code for the various tools available through the `important-sampling` package, CLI, and Boathouse toolkit. The main components are:

- **Python APIs:** Located in `src/important-sampling-watersheds-pb/` and related submodules
- **CLI Tools:** Command-line interfaces for running key workflows
- **Boathouse Integration:** Interfaces for use with the Boathouse toolkit

### Main Modules
- `src/cli/`: Command-line interface scripts
- `src/evaluation/`: Evaluation and plotting utilities
- `src/importance-sampling-watersheds-pb/`: Core importance sampling logic and modules
- `src/met/`: Meteorological data processing
- `src/toy_problem/`: Toy models and testing scripts
- `src/transpose/`: Transposition and sampling utilities
- `src/utils/`: Utility functions for geospatial and statistical operations

---

## Methodology & Workflow

The typical workflow involves:
1. Preprocessing storm catalogues
2. Sampling storm events using importance sampling
3. Computing precipitation depths and statistics
4. Evaluating and visualizing results

**Detailed workflow diagrams and flowcharts will be added in future updates.**

---

## Contributing

Please follow the coding standards and contribution guidelines linked at the top of this document. Format your code with `black` and ensure all linting and tests pass before submitting a pull request.

---

## License

This project is licensed under the terms of the LICENSE file in this repository.

---

## Contact

For questions or further information, please contact the project maintainers or refer to internal documentation.

---

# Important Sampling for SST
==============================



## **📢 REQUIRED READING**

This *README* provides crucial information for setting up and contributing to the Important-sampling project. It covers essential topics such as the dev-container setup, Python APIs, CLIs, authentication, running tests, linting, CI/CD, and formatting guidelines. If you haven't already, please read this document in it's entirety. It is necessary to ensure a smooth development process and maintain consistency across the project. If you do not understand something, reach out to someone who does!

## How to contribute

Review the following best practices for information on how to get your code merged! All code should follow the coding standards in the [Coding Standards](https://github.com/Denver-Automation-Analytics/software-design-and-best-practices/wiki/Coding-Standards). Please also review the [GitHub and Version Control](https://github.com/Denver-Automation-Analytics/software-design-and-best-practices/wiki/GitHub-and-Version-Control) wiki page. Please set up branch protection rules to enforce pull request reviews prior to merging into protected branches (especially if CI/CD is configured to run on some branches). If you require a *staging* branch, it may be helpful to set it as the default branch so pull request target that branch by default.

> Be sure your code is formatted with `black` and all linting and pytest checks pass before requesting a review of your code (see *Formatting Linting and Testing* below)

### Prerequisites

Before you can start developing, please make sure you have the following software installed on your machine:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code (VSCode)](https://code.visualstudio.com/download)
- [Remote - Containers extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setting up the Development Environment

- Make sure Docker Desktop is running.
- Clone the repository to your local machine.
- Open the project folder in VSCode.
- When prompted, click on "Reopen in Container" to open the project inside the devcontainer.
- If you don't see the prompt, you can manually open the command palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) and select "Dev Containers: Rebuild and Reopen in Container".
- Wait for the devcontainer to build and start. This may take a few minutes if it is the first time you have opened the project in a container.

### Adding dependencies

Use the `env.yaml` file at the project root directory to keep pinned dependencies up-to-date and version controlled.

> Only include top level dependencies in this file (i.e. only packages you explicity want installed and use in your code) and Only inlcude the major.minor version (this allows all patches to automatically be applied when the project is rebuilt)

If your dependencies are more complex (i.e cannot be installed / managed with micromamba alone) you may need to update the `.devcontainer/Dockerfile` and apply similar modification to the production `Dockerfile`.

## Project Organization

This repository contains the source code for the various tools available through the `important-sampling` package, cli, and Boathouse toolkit. This repository provides a Python API, CLI, and Boathouse interface to each tool.

### Python APIs

The Python API is accessed through the `important-sampling` Python package. You will find the following in the `src/` folder related to the Python API:

- `important-sampling` Python package
  - This is the main package in this repository
  - This package contains the Python APIs for each tool

