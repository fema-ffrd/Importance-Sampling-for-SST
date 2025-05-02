# Important-sampling
==============================

Michael Baker International

FEMA FFRD - Owner avatar

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

### CLIs

The CLI for each tool is defined in `src/important-sampling_cli.py`. It uses the `click` Python package. Click documentation is available [here](https://click.palletsprojects.com/en). Each new tool should have a CLI developed for it.

### Boathouse Interface

The Boathouse Interface is defined in the `interface.json` file at the root of this repository. Each new CLI should have a Boathouse Interface developed for it. A specification of this file format is not complete or stable, you can use the existing examples as a starting point for each new tool.

The `Dockerfile` at the root of the project directory (not inside the `.devcontainer` directory) is used to build the production image that is distributed through Boathouse, it may require modification depending on your requirements.

## Authentication

In order to run the tools or tests, you must be authenticated with an AWS profile that has access to develop on this repository. You can request access keys from a project developer (preferred), or sign in on the CLI with your AWS SSO credentials for the DAA AWS account.

> If you are not authenticated, all tools and tests will fail to run and report **Unauthorized** to standard output.

If using project keys for authentication. Make a new file `.env` file in the root of the project directory (this file should not be commited to source control and is ignored in the `.gitignore`). Copy the format of the variables present in the `.env.template` file and fill in the necessary keys.


## Setting up Production and Staging Environments

Boathouse tools have both *Production* and *Staging* environments. When a new repo is created, the required infrastructure needs to be configured in AWS for boathouse to pick up the tools and for CI/CD to function. Contact Brendan Barnes to set this up initially. It will require:

In Boathouse Staging AWS Account:

- An ECR repo in us-east-1 named after this toolkit
- A new entry in the DynamoDB Table *boathouse-repositories-staging*
- A new entry in the *GitHub-Staging-Role* in IAM to allow CI/CD deployments to the ECR repo and any other infrastructure required by the tool (e.g. S3 bucket)

In Boathouse Production AWS Account:

- An ECR repo in us-east-1 named after this toolkit
- A new entry in the DynamoDB Table *boathouse-repositories-production*
- A new entry in the *GitHub-Production-Role* in IAM to allow CI/CD deployments to the ECR repo and any other infrastructure required by the tool (e.g. S3 bucket)

Once the infrastructure is configured, update the `.github/workflows/build_and_push_production.yml` and the `.github/workflows/build_and_push_staging.yml` workflow files to build and push the images on merge to their respective branches.

# Formatting Linting and Testing

This project uses `black` as the Python formatter. Before you merge your code into staging or main, you should make sure all your code is formatted by running the following command from the root of the project.

```
black .
```

This project includes a linter for Python called `pylint`. Your source code will be linted in the editor. To run the linter manually, run the following command from the root directory

```
pylint --fail-under=9 src*
```

To run the project tests, open a python file and click the tests button on the extensions panel of VSCode. You may need to configure the tests with the GUI. To do this, select *Configure Tests* and choose *pytest* as the testing framework and `tests` as the testing directory.

Once the tests are loaded, you can run them with the play button in the tests extension.

Alternatively, you can run the following command from the root directory to run all tests

```
pytest
```

## CI/CD

This project incorporates a CI/CD pipeline to test and deploy updates to Boathouse. Any merge to the staging branch will build and push an update to the staging `important-sampling` repo in Boathouse. Any merge to the production branch will build and push an update to the production `important-sampling` repo in Boathouse.

Additionally, all pull requests will automatically run the tests and lint your code using the `.github/workflows/lint_and_test.yml`. If any tests or linting fails, the pull request cannot be merged.

# Running tools in development

## Notes on running command line tools during development  

Click has been selected as the command line wrapper for our projects. To run tools parameterized through click,  
Follow this example in the terminal:

```
python <path_to_python_file> <name_of_function> --<function_parameter_name> <function_parameter_value> 
```

To get the help messages:

```
python path_to_python_file --help 
```

## Notes on running command line tools once executable is built

We do not have to run this through python once built in a Docker container. You normally don't want to do this, just run it using Boathouse's GUI. If you want to run the production container, build or pull it, then:

```
docker run -v <bind mount> <image name> /dist/important-sampling_cli <command_name> --<command_arg_name> <command_arg_value> ...
```

Or from inside the container, run the tools as follows:

```
/dist/important-sampling_cli <command_name> --<command_arg_name> <command_arg_value> ...
```

For help on a particular command

```
/dist/important-sampling_cli <command_name> --help
```

For a list of all available commands

```
/dist/important-sampling_cli --help
```

## Notes on running standalone files / scripts

Normally, it is best to make a testing file under the `tests` folder to test functionality, but it is occasionally helpful to be able to run the scripts directly. In this case, you can change directories to the `src/` directory which will allow you to import files from the package or run standalone scripts that import files from the package:

```
cd src
python <- will open a python shell where "import my_package" will work
```

OR

```
cd src
python myscript.py <- the script can import code from my_package
```

## Building Dev Image

To test this image in Boathouse without pushing to staging or production branches, you can build it locally on Windows using the build-dev-image.ps1 script. First, you may need to set the execution policy

In powershell:

1. Set Execution Policy (If Needed): If you encounter an issue running scripts due to the execution policy, you can temporarily set it to unrestricted. Run the following command:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Unrestricted
```

2. Run the Script: Once you are in the correct directory, run your PowerShell script using the following command:

```
.\build-dev-image.ps1
```

This will create a local docker image for important-sampling tagged `dev` that will be available in Boathouse.

## IaC

If this repo has the `iac` directory, it contains the environment and source code for the IaC dependencies for the important-sampling package. The IaC includes an S3 bucket where documentation and other tool data is stored. The IaC code uses a separate environment (pipenv) from the micromamba environment used to develop the important-sampling package. To modify or deploy the IaC code navigate to the `iac` directory and install the pipenv with the following command:

```
cd iac
pipenv install
```

You may also need to run `cdktf get` to generate module and provider bindings. After the environment is installed, you can activate it with the following command:

```
pipenv shell
```

In order for VSCode to properly run intellisense on this code, you must change your python interpreter to this environment. To change your Python interpreter:

- Press `Ctrl+Shift+P` to open the command palette.
- Type "Python: Select Interpreter" and select it.
- VSCode will then display a list of available Python interpreters. Look for the one that corresponds to the virtual environment. It should be located within the virtual environment directory.
- Select the interpreter associated with your virtual environment.

### Deploy IaC in Development
Production and Staging deployments are managed through github actions deployments. In development, to deploy the dev infrastructure run the following command:

```
cdktf deploy important-sampling-dev
```


## Project File Structure
------------

    ├── LICENSE
    ├── README.md                <- The top-level README for developers using this project.  
    ├── .devcontainer
    │   ├── devcontainer.json    <- Dev Container specifications  
    │   ├── Dockerfile           <- Container definition for dev environment  
    │   └── README.md            <- Documentation for devcontainer  
    │
    ├── .github  
    │   └── workflows
    │       ├── build_and_push_production.yml    <- CI/CD instructions for merges into main branch (copy from another repo)
    │       ├── build_and_push_staging.yml       <- CI/CD instructions for merges into staging branch (copy from another repo)
    |       └── lint_and_test.yml                <- CI/CD instructions to lint and test code on PR
    │
    ├── data
    │   ├── 0_source       <- Source data, minimally altered from source
    │   ├── 1_interim      <- Intermediate data that has been transformed.
    │   └── 2_production   <- Fully transformed, clean datasets for next steps
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── env.yml    <- The requirements file for reproducing the analysis environment
    │
    ├── src                 
    │   ├── *repo_name                              <- Python source code for use in this project.
    │   │       ├── __init__.py                     <- Package indicator, various uses
    │   │       ├── authorizer.py                   <- Boathouse utility to perform user authentication
    │   │       ├── sp00_python_template.py         <- Example of how to structure functional programming scripts
    │   │       └── sp01_python_runner_template.py  <- Example of how to store project specific parameters
    │   │         
    │   └── *cli.py         <- The command line interface definition that is called by Boathouse
    │
    ├── test
    │   ├── __init__.py                 <- Package indicator, various uses
    │   └── test_python_template.py     <- Example of how to test functions in src folder
    │
    ├── .gitattributes      <- Handles consistent line endings in source control across multiple operating systems
    │
    ├── .gitignore          <- Handles which directories to keep out of version control
    │
    ├── .dockerignore       <- Which files to ignore when building docker image for production
    │
    ├── .gitconfig          <- Handles consistent file permissions in source control across multiple operating systems
    │
    ├── .pylintrc           <- Customizations to pylint default settings
    │
    ├── pytest.ini           <- Customizations to pytest
    │
    ├── Dockerfile          <- The production Dockerfile used for Boathouse tools
    │
    ├── interface.json      <- Defines the Boathouse GUI for each tool
    │
    ├── LICENSE             <- MBI license (copyright to MBI)
    │
    ├── README.md           <- Template information
    |
    ├── .env.template       <- Template env file (used to create a new .env file with your environment variables for authentication)
    |
    ├── build-dev-image.ps1 <- Build a dev image for Boathouse Staging / Dev local testing
    │
    └── tox.ini             <- Alternative method to store project parameters (currently not used)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
