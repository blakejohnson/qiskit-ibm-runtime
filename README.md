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

### To run the demo notebooks locally using Aer

1. Install all the requirements: `pip install -U -r requirements.txt`
2. Install the special `qiskit-ibmq-provider` branch:
  `pip install git+https://github.com/jyu00/qiskit-ibmq-provider@runtime-service` 
3. Set the environment variables:

   - `PYTHON_EXEC` needs to point to your python executable, which may live in your `virtualenv`
   or `conda` directory (e.g. `/Users/jessieyu/.pyenv/versions/qka-demo/bin/python3`)

   - `NTC_DOC_FILE` needs to point to the JSON file containing the program definition
   (e.g. `runtime/qka_doc.json`).
   
   - `NTC_PROGRAM_FILE` needs to point to the runtime program file (e.g. `runtime/qka_runtime.py`)


### To run demo notebooks using real runtime environments


#### Requirements

To use runtime via Qiskit you'll need to

- install the latest Qiskit release
- install the special `qiskit-ibmq-provider` branch:
  `pip install git+https://github.com/jyu00/qiskit-ibmq-provider@runtime-real` 
- has the `near-time-systems` role if you want to upload new programs.

#### Additional requirements for staging

- be in the `rte-test/lp-all/renierm` provider
- set the `NTC_URL` environment variable to 
`https://api-ntc.processing-staging-5dd5718798d097eccc65fac4e78a33ce-0000.us-east.containers.appdomain.cloud` 

For example:

```
import os
from qiskit import IBMQ

os.environ["NTC_URL"] = 'https://api-ntc.processing-staging-5dd5718798d097eccc65fac4e78a33ce-0000.us-east.containers.appdomain.cloud'
IBMQ.enable_account(STAGING_TOKEN, url='https://auth-dev.quantum-computing.ibm.com/api')
```

#### Additional requirements for production

- be in the `ibm-q-internal/near-time/qiskit-runtime` provider
- Set the `NTC_URL` environment variable to 
`https://api-ntc.processing-prod-5dd5718798d097eccc65fac4e78a33ce-0000.us-east.containers.appdomain.cloud` 

The only production device setup for runtime is `ibmq_montreal`.

#### Syntax

```python
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project='qiskit-runtime')

# Upload a new runtime program
program_id = provider.runtime.upload_program(name='circuit-runner', data='runtime/circuit_runner.py')
print(program_id)

# Print all available programs
provider.runtime.pprint_programs(refresh=True)

# Delete a program
provider.runtime.delete_program(program_id='id-to-delete')

# Execute a program
backend = provider.get_backend('ibmq_montreal')
runtime_inputs = {
    'circuits': circuit
}
options = {'backend_name': backend.name()}
job = provider.runtime.run(program_id="circuit-runner",
                           options=options,
                           inputs=runtime_inputs,
                           )

# See job status
print(job.status())

# Get results
result = job.result()
```

Note: Job cancel doesn't work yet.

See the [design doc](https://github.ibm.com/IBM-Q-Software/design-docs/blob/master/docs/quantum_services/Quantum_Program_Runtime/qiskit_interface.md) 
for more details.
