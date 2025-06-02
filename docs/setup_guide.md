# Environment Setup Guide

This guide provides detailed instructions for setting up the development environment for the Super Bowl Ad Analysis project.

## Virtual Environment Setup

### Windows
1. Open PowerShell as Administrator
2. Navigate to the project directory:
   ```powershell
   cd "path\to\superbowl-ad-analysis"
   ```
3. Create the virtual environment:
   ```powershell
   python -m venv .venv
   ```
4. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\activate
   ```
5. Upgrade pip and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Linux/Mac
1. Open Terminal
2. Navigate to the project directory:
   ```bash
   cd path/to/superbowl-ad-analysis
   ```
3. Create the virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
5. Upgrade pip and install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## IDE Setup

### VSCode
1. Install VSCode if not already installed
2. Install the Python extension for VSCode
3. Open the project folder in VSCode
4. Select the Python interpreter:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.venv`

### PyCharm
1. Open the project in PyCharm
2. Go to File > Settings > Project > Python Interpreter
3. Click the gear icon and select "Add"
4. Choose "Existing Environment"
5. Select the Python interpreter from `.venv`

## Jupyter Notebook Setup
1. Activate the virtual environment
2. Install Jupyter:
   ```bash
   pip install jupyter
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

## Common Issues and Solutions

### Permission Issues
If you encounter permission errors while installing packages:
1. Run PowerShell/Terminal as Administrator
2. Ensure you have write permissions to the project directory

### Virtual Environment Not Found
If the virtual environment is not found:
1. Verify the `.venv` directory exists
2. Ensure you're in the correct directory
3. Try recreating the virtual environment

### Package Installation Issues
If you have issues installing packages:
1. Ensure pip is up to date:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Try installing packages individually
3. Check for any error messages in the output

## Best Practices
1. Always activate the virtual environment before starting work
2. Keep requirements.txt updated:
   ```bash
   pip freeze > requirements.txt
   ```
3. Use version control for your code, but not for the virtual environment
4. Document any new dependencies in requirements.txt

## Additional Resources
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [VSCode Python Setup](https://code.visualstudio.com/docs/python/python-tutorial)
- [Jupyter Documentation](https://jupyter.org/documentation) 