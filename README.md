# Runtime THINK demos

This repo is used to store artifacts for the near time compute THINK demos.

### Structure of this repo:

- `qka/`
    - contains QKA source code
- `qn-spsa`
    - contains VQE source code
- `runtime/`
    - contains runtime programs
- demo notebooks are in the root directory

### Steps to run the demo notebooks using Aer

1. Install all the requirements: `pip install -U -r requirements.txt`
2. Open the notebook
3. Modify the environment variables at the top of the notebook, if necessary:

   - `PYTHON_EXEC` needs to point to your python executable, which may live in your `virtualenv`
   or `conda` directory (e.g. `/Users/jessieyu/.pyenv/versions/qka-demo/bin/python3`)

   - `NTC_DOC_FILE` needs to point to the JSON file containing the program definition
   (e.g. `runtime/qka_doc.json`).
   
   - `NTC_PROGRAM_FILE` needs to point to the runtime program file (e.g. `runtime/qka_runtime.py`)


### Steps to run demo notebooks using real runtime

If a notebook doesn't have the 3 environment variables mentioned above (`PYTHON_EXEC`, 
`NTC_DOC_FILE`, and `NTC_PROGRAM_FILE`) in the first cell, then it's already set up to use real
runtime on production.


#### Requirements

To use runtime via Qiskit you'll need to

- be in the `ibm-q-internal/near-time/qiskit-runtime` provider
- install the latest Qiskit release
- install the special `qiskit-ibmq-provider` branch:
  `pip install git+https://github.com/jyu00/qiskit-ibmq-provider@runtime-real` 
- has the `near-time-systems` role if you want to upload new programs.

The only device setup for runtime is `ibmq_montreal`.

#### Staging

If you want to use the staging system instead of production, change the `NTC_URL` environment variable
to point to staging, e.g.

```
import os
from qiskit import IBMQ

os.environ["NTC_URL"] = 'https://api-ntc.processing-staging-5dd5718798d097eccc65fac4e78a33ce-0000.us-east.containers.appdomain.cloud'
IBMQ.enable_account(STAGING_TOKEN, url='https://auth-dev.quantum-computing.ibm.com/api')
```

#### Syntax

```python
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project='qiskit-runtime')

# Upload a new runtime program
program_id = provider.runtime.upload_program(name='circuit-runner', data='runtime/circuit_runner.py')
print(program_id)

# See what's available
provider.runtime.print_programs()

# Execute a program
backend = provider.get_backend('ibmq_montreal')
runtime_params = {
    'circuits': circuit
}
options = {'backend_name': backend.name()}
job = provider.runtime.run(program_id="circuit-runner",
                           options=options,
                           params=runtime_params,
                           )

# See job status
print(job.status())

# Get results
result = job.result()
```

See the [design doc](https://github.ibm.com/IBM-Q-Software/design-docs/blob/master/docs/quantum_services/Quantum_Program_Runtime/qiskit_interface.md) 
for more details.
