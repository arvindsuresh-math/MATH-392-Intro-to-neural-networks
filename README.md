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

1. **Open VS Code**.
2. **Open a new terminal** in VS Code:
   - Go to the menu bar and select `Terminal` > `New Terminal`.

3. **Clone the repository**:
   - In the terminal, run:
     ```bash
     git clone [your-repository-url]
     cd [repository-name]
     ```

### Step 3: Create the Conda Environment

1. **Create the environment from the [environment.yml](http://_vscodecontentref_/4) file**:
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
