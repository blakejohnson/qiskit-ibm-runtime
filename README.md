# This repo is now used to store runtime programs

The `runtime/` directory contains the source code and metadata files for all the Qiskit Runtime
programs available today. They should match the files in https://github.com/Qiskit-Partners/qiskit-runtime, 
with the exceptions of

- `vqe`: Waiting for Qiskit Terra 0.18.0 release
- `circuit-runner`: Uses internal M3 measurement error mitigation 


### Using Qiskit Runtime

**Note:**

1. You need the `near-time-systems` role in order to upload a new program.
1. When updating a program, it is important to specify a `max_execution_time`, which is the maximum 
amount of time, in seconds, the program is allowed to run. If the execution time exceeds this 
number, a second runtime program will be started.


```python
from qiskit import IBMQ

provider = IBMQ.load_account()

# Upload a new runtime program.
program_id = provider.runtime.upload_program(
    name='sample-program', 
    data='runtime/sample_program.py', 
    metadata='runtime/sample_program.json'
)
print(program_id)

# Print all available programs
provider.runtime.pprint_programs(refresh=True)

# Print just one program
print(provider.runtime.program(program_id='sample-program', refresh=True))

# Delete a program
provider.runtime.delete_program(program_id='id-to-delete')

# Execute a program

def interim_result_callback(job_id, interim_result):
    print(interim_result)

runtime_inputs = {
    "iterations": 2
}
options = {'backend_name': 'ibmq_montreal'}
job = provider.runtime.run(program_id="sample-program",
                           options=options,
                           inputs=runtime_inputs,
                           callback=interim_result_callback
                          )

# See job status
print(job.status())

# Get results
result = job.result()
```
