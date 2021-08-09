import os
import glob
import sys
from qiskit import IBMQ

api_token = sys.argv[1]
IBMQ.save_account(api_token)

for file in glob.iglob("test/*/test*.py"):
    os.system("python " + file)
