# MATH 392 Intro to neural networks
 Parent repo for topics course taught at U of Arizona in Spring 2025.

To determine the system, Python, and package requirements for your neural networks course repository, we need to consider the typical tools and libraries used for neural network tasks. Here are some common requirements:

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **RAM**: At least 8GB (16GB recommended)
- **Disk Space**: At least 10GB free space
- **GPU**: Optional but recommended for training large models (NVIDIA GPU with CUDA support)

### Python Requirements
- **Python Version**: Python 3.8 or higher

### Package Requirements
Here are some common Python packages used for neural network tasks:

1. **NumPy**: For numerical computations
2. **Pandas**: For data manipulation and analysis
3. **Matplotlib**: For plotting and visualization
4. **Scikit-learn**: For machine learning utilities
5. **TensorFlow**: For building and training neural networks
6. **Keras**: High-level neural networks API, running on top of TensorFlow
7. **PyTorch**: An alternative to TensorFlow for building and training neural networks
8. **Jupyter**: For running Jupyter notebooks
9. **SciPy**: For scientific computing
10. **Seaborn**: For statistical data visualization

### Example `requirements.txt`
Here is an example `requirements.txt` file that includes the above packages:

```
numpy>=1.19.2
pandas>=1.1.3
matplotlib>=3.3.2
scikit-learn>=0.23.2
tensorflow>=2.3.0
keras>=2.4.3
torch>=1.7.0
jupyter>=1.0.0
scipy>=1.5.2
seaborn>=0.11.0
```

### Example 

environment.yml

 for Conda
If you prefer using Conda, here is an example 

environment.yml

 file:

```yaml
name: neural_networks_course
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - numpy>=1.19.2
  - pandas>=1.1.3
  - matplotlib>=3.3.2
  - scikit-learn>=0.23.2
  - tensorflow>=2.3.0
  - keras>=2.4.3
  - pytorch>=1.7.0
  - jupyter>=1.0.0
  - scipy>=1.5.2
  - seaborn>=0.11.0
```

### Setting Up the Environment
To set up the environment using `requirements.txt`:
```sh
pip install -r requirements.txt
```

To set up the environment using 

environment.yml

:
```sh
conda env create -f environment.yml
conda activate neural_networks_course
```

These requirements should cover the typical needs for practical coding tasks related to neural networks. Adjust the versions and packages as necessary based on the specific tasks and tools you plan to use in your course.