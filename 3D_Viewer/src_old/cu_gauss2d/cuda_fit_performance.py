#
#  file:  cuda_fit_performance.py
#
#  Call IDL repeatedly, quitting each time to reset the state,
#  to determine the performance as a function of the number of images.
#
#  RTK, 23-Nov-2009
#  Last update:  23-Nov-2009
#
###############################################################

import sys
import os

def main():
    """Call IDL repeatedly passing the number of images to use"""

    starting  =      100
    ending    = 10000000
    increment =    10000

    i = starting
    while (i <= ending):
        os.system("idl -quiet -e cuda_fit_performance -args %d" % i)
        i += increment
        
if __name__ == "__main__":
    main()

