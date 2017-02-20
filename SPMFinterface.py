import subprocess
from datetime import datetime as dt

def callSPMF(jarpath,algorithm,sequenceDbPath,outputPath,minSup,minConf):
    command = ' '.join(['java -jar',spmfJarPath,'run',algorithm,sequenceDatabasePath,outputPath,minsup,minconf])

    p = subprocess.Popen(command,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
    p.communicate() # wait for completion


spmfJarPath = 'C:/Users/Umberto/Desktop/seq_rec_evaluation/spmf/spmf.jar'
algorithm = 'RuleGrowth'
#sequenceDatabasePath = 'C:/Users/Umberto/Desktop/seq_rec_evaluation/spmf/contextPrefixSpan.txt'

listOfDBs = ['contextPrefixSpan.txt','SIGN.txt','LEVIATHAN.txt','BMS1_spmf.txt','MSNBC.txt','BMS2.txt','Kosarak_converted.txt','BIBLE.txt','MSNBC_SPMF.txt']
sequenceDatabasePath = 'datasets/Kosarak_converted.txt'
outputPath='output.txt'
minconf='1%'
minsup='1%'

tic = dt.now()
callSPMF(spmfJarPath,algorithm,sequenceDatabasePath,outputPath,minsup,minconf)
print('Execution time is {}'.format(dt.now()-tic))

for db in listOfDBs:
    tic = dt.now()
    callSPMF(spmfJarPath,algorithm,'datasets/'+db,outputPath,minsup,minconf)
    print('Execution time of {} is {} minsup={} minconf={}'.format(db,dt.now()-tic,minsup,minconf))


