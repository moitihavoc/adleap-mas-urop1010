# Importing the necessary packages
from os.path import isdir
import subprocess
import time

# Defining supportive methods
def python_cmd(env,method,exp_type,id=0,mode='default'):
    return "python examples/"+env+"_"+exp_type+".py"+\
                                " --atype "+method+\
                                " --exp_num "+str(i)+\
                                " --id "+str(id)+\
                                " --mode "+str(mode)

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
envs = ['tiger','maze','rocksample','levelforaging']#['tiger','maze','rocksample','levelforaging','tag','lasertag']
methods = ['pomcp','tbrhopomcp']#,'iprpomcp','iucbpomcp','libpomcp','ibpomcp']#['pomcp','tbrhopomcp','ibpomcp','rhopomcp','ipftreed']
nexperiments = 50
scenario_id = {
    'tiger'         :[0],
    'maze'          :[2],#[0,1,2,3],
    'rocksample'    :[2],#[0,1,2,3],
    'levelforaging' :[4],#[0,1,2,3,4],
    #'tag'           :[0],
    #'lasertag'      :[0]
}
mode_list = ['default']

# Lunching experiments
for env in envs:
    for i in range(0,nexperiments):
        for method in methods:
            for mode in mode_list:
                for id in scenario_id[env]:
                    with open('run.sh','w') as runfile:
                        runfile.write("#!/bin/bash\n\n")
                        runfile.write("#SBATCH -J "+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"\n")
                        runfile.write("#SBATCH -e serror/"+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"%j.err\n")
                        runfile.write("#SBATCH -o soutput/"+method+"_"+mode+'_'+env+str(id)+"_"+str(i)+"%j.out\n")
                        runfile.write("#SBATCH --mem=4G\n")
                        runfile.write("#SBATCH --time=01:00:00\n")
                        runfile.write("source /etc/profile\n")
                        command = python_cmd(env, method, exp_type, id, mode)
                        runfile.write(
                            "python profile_runner.py "
                            f'--method {method} '
                            f'--env {env} '
                            f'--id {id} '
                            f'--exp {i} '
                            f'--command "{command}"\n'
                        )
            
                    subprocess.run(["sbatch","run.sh"])
                    time.sleep(0.1)