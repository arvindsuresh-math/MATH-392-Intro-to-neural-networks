# MATH 392: Intro to Neural Networks
Parent repo for topics course taught at U of Arizona in Spring 2025.

## Table of Contents

- [Course Details](#course-details)
- [Course Overview](#course-overview)
- [Weekly Topics](#weekly-topics)
- [Setup](#setup)
  - [Setup your own repository](#setup-your-own-repository)
  - [Create the Conda Environment](#create-the-conda-environment)
- [Managing your fork](#managing-your-fork)
  - [Updating from the parent](#updating-from-the-parent)
  - [Making changes to your fork](#making-changes-to-your-fork)
- [Tips for using Github Copilot](#tips-for-using-github-copilot)

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

## Setup

### Setup your own repository

1. **Fork the repository**: On the MATH 392 repo page on github, click on the `Fork` button at the top right corner of the page. This will create a copy of the repo (called a *fork*) under your GitHub account. You can make changes to your fork without affecting the parent repo (which I will be making changes to).

2. **Clone the repository**: Next, you need to "clone" your fork, which means making a local copy of the repo on your machine (desktop or laptop). The easiest way to do this is using GitHub Desktop:
   - Download and install [GitHub Desktop](https://desktop.github.com/).
   - Open GitHub Desktop and sign in to your GitHub account.
   - Click on `File` > `Clone repository`.
   - In the `URL` tab, paste the URL *of your fork* and choose the local path where you want to clone the repository, and click `Clone`.

3. **Add the parent repo as a remote repo**: Next, you need to add the parent repo as a *remote repo* so that you can update your fork whenever I add or make changes to the parent. For this, open a terminal in VS Code and run:
```bash
git remote add parent https://github.com/arvindsuresh-math/MATH-392-Intro-to-neural-networks.git
```
4. **Create a branch for work**: You will often need to add/make changes to your fork. Best practice is to always create a separate *branch* to make changes, and then merge these changes into your `main` branch. For simplicity, I suggest you make only one separate branch at the start of the semester:
   - In GitHub Desktop, click on `Current Branch` and select `New Branch`.
   - Name your branch (e.g., `myname-work`) and click `Create Branch`.

### Create the Conda Environment

We will be using a bunch of different packages to write code in this course. We will install them in one fell swoop by creating a Conda Environment for our course, as follows. 

1. **Download and install Miniconda**:
   - Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
   - Download the installer for your operating system (Windows or macOS), and run it to complete the installation.

2. **Initialize Conda**: Open the terminal in VS Code and run: 
   ```bash
   conda init
   ```
   Then, close and re-open the terminal to apply the changes. 

3. **Create the environment from the [environment.yml](http://_vscodecontentref_/1) file**:
   - Open the repository in VS Code, and open a new terminal window.
   - In the terminal, first run:
      ```bash
      conda init
      ```
   - Close and re-open the terminal to apply the changes, and then run:
      ```bash
      conda env create -f environment.yml
      ```
   - Finally, to activate the environment, run:
      ```bash
      conda activate math392
      ```
   - To verify that the setup worked as expected, open the Jupyter notebook `environment_check.ipynb` in the `Getting started` folder and follow the instructions.

## Managing your fork

### Updating from the parent

I will regularly be adding/changing changes to the parent repo, and it is important to always work with the up-to-date version. Whenever you sit down to work on VS Code, **always start by doing the following**:

1. Open your fork in GitHub Desktop, and make sure that your current branch is set to `main`.

2. Click on `Fetch origin` to fetch the latest changes from the parent repo. 

2. Click on `Branch` > `Merge into current branch` (select `parent/main`) to merge the changes from the parent repo into your local `main` branch.

3. Click on `Push origin` to push the changes to your fork on GitHub (more precisely, to the `main` branch on your fork).

### Making changes to your fork

You will often need to add/make changes to your fork. Remember, every time you sit down to work in VS Code, first update from the parent repo as outline above. Then, do the following:

1. Open your fork in GitHub Desktop, and make sure that your current is set to `myname-work` (i.e. the branch you made for your changes).

2. On VS Code, the bottom-left corner will show the current branch you are working it. Now, go ahead and make your changes to the files in your local repo. These changes are known only to the current branch `myname-work`.

3. Save your changes (called **committing the changes**) as follows:
   - In GitHub Desktop, you will see the changed files listed in the `Changes` tab.
   - Add a summary of the changes you made in the `Summary` box.
   - Click `Commit to myname-work`.

4. Click on `Push origin` to push your changes to your fork on GitHub (more precisely, to the `myname-work` branch on your fork).

5. Finally, to merge these changes into the `main` branch of your fork, click on `Create pull request` on GitHub Desktop. 

## Tips for using Github Copilot

If you want to make use of Github Copilot, there are three ways:
1. **For short, simple tasks requiring no explanation**: Write a comment explaining (in natural language) what you want to accomplish, and hit enter. When you start typing in the next line (or after a few moments), copilot will give auto-complete suggestions that you can accept by hitting tab.

2. **For somewhat longer tasks, still needing no explanation**:
Start an inline chat with copilot and ask it to generate code to accomplish what you want. Copilot will generate code in place and typically include comments explaining what each block is doing.

3. **For long, complex tasks requiring explanation**: 
Use the separate copilot chat to write your prompt. Copilot will automatically explain the code it generates for you (in addition to comments for each code block).

