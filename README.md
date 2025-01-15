# MATH 392: Intro to Neural Networks
Parent repo for topics course taught at U of Arizona in Spring 2025.

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
     git clone https://github.com/arvindsuresh-math/MATH-392-Intro-to-neural-networks
     cd MATH-392-Intro-to-neural-networks
     ```

2. **Using GitHub Desktop**:
   - Download and install [GitHub Desktop](https://desktop.github.com/).
   - Open GitHub Desktop and sign in to your GitHub account.
   - Click on `File` > `Clone repository`.
   - In the `URL` tab, paste the repository URL [MATH-392-Intro-to-neural-networks](https://github.com/arvindsuresh-math/MATH-392-Intro-to-neural-networks) and choose the local path where you want to clone the repository.
   - Click `Clone`.

### Step 3: Create the Conda Environment

1. **Create the environment from the [environment.yml](http://_vscodecontentref_/1) file**:
   - In the terminal, run:
     ```bash
     conda env create -f environment.yml
     ```

2. **Activate the environment**:
   - In the terminal, run:
     ```bash
     conda activate math392
     ```

### Step 4: Verify the Installation

1. **Verify that the environment is set up correctly**:
   - In the terminal, run:
     ```bash
     python -c "import torch; print(torch.__version__)"
     ```

### Step 5: Running Jupyter Notebooks in VS Code

1. **Open the terminal in VS Code**:
   - Go to the menu bar and select `Terminal` > `New Terminal`.

2. **Activate the Conda environment** (if not already activated):
   - In the terminal, run:
     ```bash
     conda activate math392
     ```

3. **Open the Jupyter Notebook in VS Code**:
   - In the VS Code Explorer, navigate to the directory containing the Jupyter notebook you want to open.
   - Click on the notebook file (`.ipynb`) to open it.
   - VS Code will automatically open the notebook in the Jupyter Notebook interface within the editor.

4. **Run the notebook cells**:
   - You can run the cells in the notebook by clicking the `Run` button or by pressing `Shift + Enter`.

