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

