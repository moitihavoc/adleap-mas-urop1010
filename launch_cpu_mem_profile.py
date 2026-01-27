"""
File to lunch examples/*.py experiments on a HEC cluster
"""
from src.utils import monitoring

# Defining supportive methods
def adleapmas_cmd(env,method,exp_type,id=0,mode='default'):
    return "examples/"+env+"_"+exp_type+".py"+\
                            " --atype "+method+\
                            " --exp_num "+str(i)+\
                            " --id "+str(id)+\
                            " --mode "+str(mode)

# Setting experiments configuration
exp_type = 'smalltest'
env = "maze"
methods = ['pomcp','ibpomcp','tbrhopomcp','rhopomcp']
nexperiments = 50
scenario_id = [2]
mode = 'default'

# Lunching experiments
for method in methods:
    for id in scenario_id:
        for i in range(0,nexperiments):
            py_cmd = adleapmas_cmd(env,method,exp_type,id,mode)
            full_cmd = "python "+py_cmd
            monitoring.cpu_and_mem(full_cmd,method,env+str(id),i)