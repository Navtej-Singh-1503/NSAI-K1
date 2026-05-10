import sys
import time

def slowprint(s):
        for c in s + '\n' :
                sys.stdout.write(c)
                sys.stdout.flush()
                time.sleep(2. / 100)