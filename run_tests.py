import os
import glob
import sys

for file in glob.iglob("test/*/test*.py"):
    os.system("python " + file)
