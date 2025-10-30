# ryan
Repository to host a pipeline for electrocatalysis CO2 reduction data analysis and ML

## Development Setup (Using Conda + uv)

This project uses **uv** to manage dependency groups defined in `pyproject.toml`.  
If you want to work on the codebase in editable mode while using **Conda** for the Python environment, follow the steps below.


1. Create a new Conda environment (modify the name if desired)
This sets up an isolated Python environment.
```bash
conda create -n ryan python=3.12 
```
2. Activate the environment
```bash
conda activate ryan
```

3. Install uv
uv is used to read dependency groups and resolve dependencies.
```bash
pip install uv
```

4. Install the project in editable mode with optional dependency groups
    --group analysis: Installs dependencies for data analysis workflows
    --group ml_pipeline: Installs dependencies for the ML pipeline components
```bash
uv pip install --group analysis --group ml_pipeline -e .
```

5. (Optional) Install development tools (linting, testing, formatting, etc.)
```bash
uv pip install --group dev -e .
```

6. (Optional) Register this Conda environment as a Jupyter kernel
    This allows you to select the kernel in VS Code / JupyterLab / notebooks.
```bash
python -m ipykernel install --user --name ryan --display-name "ryan"
```

