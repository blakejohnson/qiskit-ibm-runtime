# ntc-ibm-programs

This repo is now used to store runtime programs (primitives and non-primitives).

The `programs` directory contains the source code and metadata files for all **public** Qiskit Runtime
programs.

## Creating a Qiskit Runtime program

Refer to the tutorials in https://github.com/Qiskit/qiskit-ibm-runtime/tree/main/docs/tutorials
on how to create and upload your own Qiskit Runtime programs. The `sample_vqe_program` directory
contains an in-depth walk through on creating a more realistic program.

All non-open users can upload runtime programs now, and new runtime programs are by default private.
A private program can only be seen and used by its owner. Therefore you can upload and
test your program before opening a PR here.

## Publishing a Qiskit Runtime program

Once you are happy with your program and want to make it public for all to see/use, simply open
a PR in this repo. But note that

- The source script and metadata file of your program must be named `<program_name>.py` and
`<program_name>.json`, respectively, where `<program_name>` is the name of your program (but with
 underscore). Make sure the `name` field in the metadata matches your program name.

    - Python module names cannot have hyphens, and runtime program IDs cannot have underscores. Therefore,
    we have been using `program_name.py` to name the source file and `program-name` as the program name.

- Remember to add test cases for your program. Qiskit Runtime always updates to the latest Qiskit, and
the nightly CI run for this repo will hopefully catch any incompatibilities early.

    - Runtime now appends a random suffix to the program name to make the program ID. For example, if
    you create a program named `hello-world`, the program ID will be something like `hello-world-ro3PBBMNB9`.
    Therefore your test case will fail if you try to do `runtime.run(program_id="hello-world")`.

        Instead, you can use the [find_program_id](https://github.ibm.com/IBM-Q-Software/ntc-ibm-programs/blob/master/test/integration/utils.py#L4)
        utility function to find the correct ID:

        ```python
        from .utils import find_program_id
        program_id = find_program_id(self.provider.runtime, "hello-world")
        job = self.provider.runtime.run(program_id, ...)
        ```

- Once the PR is merged, the program is uploaded and made public on staging.
The CI will also fix the program ID to make it pretty (e.g. changing `hello-world-ro3PBBMNB9` to `hello-world`).

- In order to deploy the changes to production a release tag has to be created.

The same process can be used to update an existing public program.

## Quickstart on using Qiskit Runtime

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# Upload a new runtime program.
program_id = service.upload_program(
    data='programs/hello_world.py',
    metadata='programs/hello_world.json'
)
print(program_id)

# Print all available programs
service.pprint_programs(refresh=True)

# Print just one program
print(service.program(program_id='hello-world', refresh=True))

# Delete a program
service.delete_program(program_id='id-to-delete')

# Execute a program
def result_callback(job_id, result):
    print(result)

runtime_inputs = {
    "iterations": 2
}
options = {'backend_name': 'ibmq_montreal'}
job = service.run(program_id="hello-world",
                           options=options,
                           inputs=runtime_inputs,
                           callback=result_callback
                          )

# See job status
print(job.status())

# Get results
result = job.result()
print(result)
```
