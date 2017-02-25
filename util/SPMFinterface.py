import subprocess
from datetime import datetime as dt

def callSPMF(spmfPath,command):
    #java -jar spmf.jar run PrefixSpan contextPrefixSpan.txt output.txt 50%
    comm = ' '.join(['java -jar',spmfPath,'run',command])
    print(comm)
    p = subprocess.Popen(comm,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
    p.communicate() # wait for completion


def testPerformances():
    spmfJarPath = 'C:/Users/Umberto/Desktop/seq_rec_evaluation/spmf/spmf.jar'
    algorithm = 'RuleGrowth'
    #sequenceDatabasePath = 'C:/Users/Umberto/Desktop/seq_rec_evaluation/spmf/contextPrefixSpan.txt'

    listOfDBs = ['contextPrefixSpan.txt','SIGN.txt','LEVIATHAN.txt','BMS1_spmf.txt','MSNBC.txt','BMS2.txt','Kosarak_converted.txt','BIBLE.txt','MSNBC_SPMF.txt']
    sequenceDatabasePath = 'datasets/contextPrefixSpan-simplified.txt'
    outputPath='output.txt'
    minconf='50%'
    minsup='50%'

    tic = dt.now()
    command = ' '.join([algorithm,sequenceDatabasePath,outputPath,minsup,minconf])
    callSPMF(spmfJarPath,command)
    print('Execution time is {}'.format(dt.now()-tic))

    for db in listOfDBs:
        tic = dt.now()
        command = ' '.join([algorithm,'datasets/'+db,outputPath,minsup,minconf])
        callSPMF(spmfJarPath,command)
        print('Execution time of {} is {} minsup={} minconf={}'.format(db,dt.now()-tic,minsup,minconf))


