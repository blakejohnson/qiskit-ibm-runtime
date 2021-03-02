# Qiskit Kernel Alignment demo

This repo is used to store artifacts for the near time compute THINK demo with Quantum Kernel Alignment.

### Structure of this repo:

- `qka/`
    - contains QKA source code
- `runtime/`
    - contains runtime program
- demo notebooks are in the root directory

### Steps to run the demo notebooks

1. Install all the requirements: `pip install -U -r requirements.txt`
2. Open the notebook
3. Modify the environment variables at the top of the notebook, if necessary:

   - `PYTHON_EXEC` needs to point to your python executable, which may live in your `virtualenv`
   or `conda` directory (e.g. `/Users/jessieyu/.pyenv/versions/qka-demo/bin/python3`)

   - `NTC_DOC_FILE` needs to point to the JSON file containing the program definition
   (e.g. `runtime/qka_doc.json`).
   
   - `NTC_PROGRAM_FILE` needs to point to the runtime program file (e.g. `runtime/qka_runtime.py`)
