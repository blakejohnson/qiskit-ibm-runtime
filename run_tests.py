# for f in test/*test*.py; 
# do python3 "$f"; done

import os
import glob

for file in glob.iglob("test/*/test*.py"):
    os.system("python3 " + file)