import subprocess
import time
import psutil

def cpu_and_mem(command,method,env,expn):
    process = subprocess.Popen(command,shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
    print('PID:',process.pid,'EXP:',method+'_'+env+'_'+str(expn))

    file = open('./cpu_mem_profile/'+method+'_'+env+'_'+str(expn)+'.csv','w+')
    file.write("Time;CPU;RAM\n")
    start_time = time.time()
    p = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            file.write("%.4f;%.4f;%.4f\n" % \
                (time.time()-start_time,p.cpu_percent(),p.memory_percent()))
        except:
            print('Finished this process')
        time.sleep(0.05)
    file.close()