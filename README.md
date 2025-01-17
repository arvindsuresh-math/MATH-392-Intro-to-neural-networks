# MATH 392: Intro to Neural Networks
Parent repo for topics course taught at U of Arizona in Spring 2025.

## Table of Contents

- [Course Details](#course-details)
- [Course Overview](#course-overview)
- [Weekly Topics](#weekly-topics)
- [Environment Setup](#environment-setup)
   - [Step 1: Install Miniconda](#step-1-install-miniconda)
   - [Step 2: Clone the Repository](#step-2-clone-the-repository)
   - [Step 3: Create the Conda Environment](#step-3-create-the-conda-environment)
   - [Step 4: Running Jupyter Notebooks in VS Code](#step-4-running-jupyter-notebooks-in-vs-code)

## Course Details

**Instructor**: Arvind Suresh ([arvindsuresh@arizona.edu](arvindsuresh@arizona.edu))

**Credits**: 3 (counts toward the ‘Application Course’ requirement for math majors)

**Prerequisites**: MATH 223 
- Can potentially be
(Note: more or less time will be spent as needed to cover the requisite mathematical content, the pre-requisites are mainly to ensure a reasonable pace can be maintained.)

**Coding prereqs**: Experience with Python is useful but not needed because a lot of code will already be provided, and we will make systematic use of Github Copilot to get by with minimal coding.


**When**: Mon-Wed, 11 – 12:15 pm

**Where**: Modern Languages, Rm 201.

## Course Overview
This course aims to provide students with a self-contained introduction to the mathematics and practical implementation of neural networks, which are a fundamental class of machine learning models that underlie modern AI’s like ChatGPT. 

**Learning Objectives**:
- Understand the key mathematical concepts used in neural networks, including linear algebra, gradient descent, and backpropagation.
- Learn to build and implement simple neural networks using libraries like PyTorch.
- Analyze and evaluate neural network models, with an emphasis on model optimization, regularization, and hyperparameter tuning.
- Gain experience in the research method (namely, asking questions and being able to hunt down answers or resources).
- Prepare for independent research by developing the ability to approach problems related to neural networks and machine learning with a solid mathematical framework.

## Weekly Topics

- **Week 1**: Introduction to Machine Learning and Neural Networks
- **Week 2**: Mathematical Foundations – Linear Algebra (I)
- **Week 3**: Mathematical Foundations – Basic Statistics
- **Week 4**: Mathematical Foundations – Basic Probability
- **Week 5**: Introduction to Loss Functions in Machine Learning
- **Week 6**: Linear Regression and Its Connection to Neural Networks
- **Week 7**: The Perceptron – Concepts, History, and the XOR Problem
- **Week 8**: Introduction to Multilayer Perceptrons (MLPs)
- **Week 9**: Introduction to Optimization via Gradient Descent
- **Week 10**: Backpropagation
- **Week 11**: Activation Functions
- **Week 12**: Regularization in Neural Networks
- **Week 13**: Model Evaluation and Hyperparameter Tuning

## Environment Setup

### Step 1: Install Miniconda

1. **Download Miniconda**:
   - Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
   - Download the installer for your operating system (Windows, macOS, or Linux).

2. **Install Miniconda**:
   - Follow the installation instructions for your operating system:
     - **Windows**: Run the downloaded `.exe` file and follow the prompts.
     - **macOS**: Open the downloaded `.pkg` file and follow the prompts.
     - **Linux**: Open a terminal, navigate to the directory where you downloaded the installer, and run:
       ```bash
       bash Miniconda3-latest-Linux-x86_64.sh
       ```

### Step 2: Clone the Repository

1. **Using the Terminal**:
   - Open a new terminal in VS Code:
     - Go to the menu bar and select `Terminal` > `New Terminal`.
   - In the terminal, run:
     ```bash
     git clone https://github.com/your-username/your-repository-name.git
     cd your-repository-name
     ```

2. **Using GitHub Desktop**:
   - Download and install [GitHub Desktop](https://desktop.github.com/).
   - Open GitHub Desktop and sign in to your GitHub account.
   - Click on `File` > `Clone repository`.
   - In the `URL` tab, paste the repository URL and choose the local path where you want to clone the repository.
   - Click `Clone`.

### Step 3: Create the Conda Environment

1. **Activate Conda**:
   - Open the terminal in VS Code.
   - If Conda is not already initialized, run:
     ```bash
     conda init
     ```
   - Close and reopen the terminal to apply the changes.

2. **Create the environment from the [environment.yml](http://_vscodecontentref_/1) file**:
   - Open the repository in VS Code.
   - In VS Code, open a new terminal window.
   - In the terminal, run:
     ```bash
     conda env create -f environment.yml
     ```

3. **Activate the environment**:
   - In the terminal, run:
     ```bash
     conda activate math392
     ```

4. **Verify that the environment is set up correctly**:
   - In the terminal, run:
     ```bash
     python -c "import torch; print(torch.__version__)"
     ```

### Step 4: Running Jupyter Notebooks in VS Code

1. **Open the terminal in VS Code**:
   - Go to the menu bar and select `Terminal` > `New Terminal`.

2. **Activate the Conda environment** (if not already activated):
   - In the terminal, run:
     ```bash
     conda activate math392
     ```

3. **Open the Jupyter Notebook in VS Code**:
   - In the VS Code Explorer, navigate to the directory containing the Jupyter notebook (`.ipynb`)  and click (or double-click) on the file to open it. 
   - VS Code will automatically open the notebook in the Jupyter Notebook interface within the editor.

4. **Run the notebook cells**:
   - You can run the cells in the notebook by clicking the `Run` button or by pressing `Shift + Enter`. When you do this the first time, you will be asked to select a "kernel". The environment 
   `math392` should be available as one of the options; select this and then run your cells.
