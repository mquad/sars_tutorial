import pandas as pd
from collections import Counter
import datetime
import calendar
import time

def createSeqDbList():
    filePath = 'C:/Users/Umberto/Dropbox/datasets/music/sessions.csv'
    file = pd.read_csv(filePath)

    #group by session id and concat song_id
    groups = file.groupby('session_id')

    aggregated = groups['song_id'].agg({'songs':lambda x:list(x)})
    initialTimestamps = groups['ts'].min()

    result = aggregated.join(initialTimestamps)
    result.to_csv('C:/Users/Umberto/Dropbox/datasets/music/sequenceDb.csv',index=False)

def createSeqDbCommas():
    filePath = 'C:/Users/Umberto/Dropbox/datasets/music/sessions.csv'
    file = pd.read_csv(filePath)

    groups = file.groupby('session_id')
    initialTimestamps = groups['ts'].min()

    file['song_id'] = file['song_id'].astype(str)
    agg=groups['song_id'].apply(lambda x: "%s" % ', '.join(x))
    result = pd.DataFrame(agg).join(initialTimestamps)

    result.to_csv('sequenceDbComma.csv',index=False)

def create_seq_db_filter_top_k(filePath, topk, last_months):
    file = load_and_adapt(filePath,last_months)

    c = Counter(list(file['item_id']))

    keeper = set([x[0] for x in c.most_common(topk)])
    file = file[file['item_id'].map(lambda x: x in keeper)]

    #group by session id and concat song_id
    groups = file.groupby('session_id')

    aggregated = groups['item_id'].agg({'sequence':lambda x:list(x)})
    initialTimestamps = groups['ts'].min()
    users = groups['user_id'].min() #it's just fast, min doesn't actually make sense

    result = aggregated.join(initialTimestamps).join(users)
    return result
    #result.to_csv('sequenceDb_'+str(topk)+'.csv',index=False)


def from_db_to_spmfdb(filePath,last_months):
    file = load_and_adapt(filePath,last_months)

    groups = file.groupby('session_id')
    #songs as strings

    file['item_id'] = file['item_id'].astype(str)
    agg=groups['item_id'].apply(lambda x: "%s" % ', '.join(x))

    outputFile = 'sequences.txt'
    with open(outputFile,'w') as fout:
        for i,row in pd.DataFrame(agg).iterrows():
            fout.write(row['item_id'].replace(", ", " -1 "))
            fout.write(' -2\n')

    return outputFile

def from_seqs_to_spmfdb(seqs):
    outputFile = 'sequences.txt'
    with open(outputFile,'w') as fout:
        for s in seqs:
            fout.write(' -1 '.join(s))
            fout.write(' -2\n')

    return outputFile

def reformat_for_SPMC(data):
    train_data_supervised = []
    for i,row in data.iterrows():
        train_data_supervised.append((int(row['user_id']),int(row['sequence'][len(row['sequence'])-1]),list(map(int,row['sequence'][:len(row['sequence'])-1]))))


def load_and_adapt(filePath,last_months):
    file = pd.read_csv(filePath)

    col_names = ['session_id','user_id','item_id','ts']+ file.columns.values.tolist()[4:]
    file.columns =col_names

    #sottocampiona
    def add_months(sourcedate,months):
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12 )
        month = month % 12 + 1
        day = min(sourcedate.day,calendar.monthrange(year,month)[1])
        return datetime.date(year,month,day)

    lastdate = datetime.datetime.fromtimestamp(int("1421745663"))
    firstdate = add_months(lastdate,-last_months)
    initial_unix = time.mktime(firstdate.timetuple())

    #filter out older items
    file = file[file['ts'] >= initial_unix]

    return file
