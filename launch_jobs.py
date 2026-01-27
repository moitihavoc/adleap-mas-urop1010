"""
File to lunch examples/*.py experiments on a HEC cluster
"""
# Importing the necessary packages
from os.path import isdir
import subprocess
import time

# Defining supportive methods
def python_cmd(env,method,exp_type,id=0,mode='default'):
    if id != 'random':
        return "python examples/"+env+"_"+exp_type+".py"+\
                                " --atype "+method+\
                                " --exp_num "+str(i)+\
                                " --id "+str(id)+\
                                " --mode "+str(mode)
    else:
        return "python examples/"+env+"_"+exp_type+".py"+\
                                " --atype "+method+\
                                " --exp_num "+str(i)+\
                                " --id "+str(id)+\
                                " --mode "+str(mode)+\
                                " --display False"+\
                                " --nagents 5"+\
                                " --ntasks 30"

# Checking AdLeap-MAS folder integrity
if not isdir("results"):
    subprocess.run(["mkdir","results"])
if not isdir("serror"):
    subprocess.run(["mkdir","serror"])
if not isdir("soutput"):
    subprocess.run(["mkdir","soutput"])
if not isdir("tmp"):
    subprocess.run(["mkdir","tmp"])

# Setting experiments configuration
exp_type = 'smalltest'
env = "levelforaging"
methods = ['ibpomcp']
nexperiments = 50
scenario_id = [5, 6, 7, 8, 9, 10, 11, 12, 13]
mode_list = ['default']

# Lunching experiments
for i in range(0,nexperiments):
    for method in methods:
        for mode in mode_list:
            for id in scenario_id:
                with open('run.sh','w') as runfile:
                    runfile.write("#!/bin/bash\n\n")
                    runfile.write("#SBATCH -J "+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"\n")
                    runfile.write("#SBATCH -e serror/"+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"%j.err\n")
                    runfile.write("#SBATCH -o soutput/"+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"%j.out\n")
                    runfile.write("#SBATCH --mem=3G\n")
                    runfile.write("#SBATCH --time=03:00:00\n")
                    runfile.write("source /etc/profile\n")
                    runfile.write(python_cmd(env,method,exp_type,id,mode))
        
                subprocess.run(["sbatch","run.sh"])
                time.sleep(0.1)