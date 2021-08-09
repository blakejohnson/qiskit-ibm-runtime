from qiskit import IBMQ

provider = IBMQ.load_account()
input = {
    "iterations": 2
}

options = {'backend_name': 'ibmq_qasm_simulator'}

def interim_result_callback(job_id, interim_result):
    print("Interim result:", job_id, interim_result)

job = provider.runtime.run(program_id="sample-program",
                           options=options,
                           inputs=input,
                           callback=interim_result_callback
                          )

print("Runtime:", job.result())
