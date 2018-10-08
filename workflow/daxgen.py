#!/usr/bin/env python

import sys
import os
import pwd
import time
from Pegasus.DAX3 import *

# The name of the DAX file is the first argument
if len(sys.argv) != 2:
   sys.stderr.write("Usage: %s DAXFILE\n" % sys.argv[0])
   sys.exit(1)

daxfile = sys.argv[1]
USER = pwd.getpwuid(os.getuid())[0]

dax = ADAG("hic_wf")
dax.metadata("name", "HIC")
dax.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dax.metadata("created", time.ctime())

events_exec = Executable("run_events")

# if you need multiple runs just add the job in a for loop
# replace XYZ with the unique identifier
# and change the name of the input and args files
# eg.

args_conf = File("args.conf")
results_in = File("Results.tar.gz")

for i in range(10): 
    results_out = File("Results_"+str(i)+".tar.gz")

    hic_job = Job("run_wrapper")
    hic_job.addArguments(args_conf, str(i))
    hic_job.uses(events_exec, link=Link.INPUT)
    hic_job.uses(args_conf, link=Link.INPUT)
    hic_job.uses(results_in, link=Link.INPUT)
    hic_job.uses(results_out, link=Link.OUTPUT, transfer=True, register=False)
    dax.addJob(hic_job)

# end of loop

f = open(daxfile,"w")
dax.writeXML(f)
f.close()
print "Generated dax %s" % daxfile
