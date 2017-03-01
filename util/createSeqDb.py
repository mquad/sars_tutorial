import pandas as pd
from collections import Counter

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

def createSeqDb_filter_top_k(k):
    filePath = 'C:/Users/Umberto/Dropbox/datasets/music/sessions.csv'
    file = pd.read_csv(filePath)

    c = Counter(list(file['song_id']))
    keeper = set([x[0] for x in c.most_common(k)])
    file = file[file['song_id'].map(lambda x: x in keeper)]

    #group by session id and concat song_id
    groups = file.groupby('session_id')

    aggregated = groups['song_id'].agg({'songs':lambda x:list(x)})
    initialTimestamps = groups['ts'].min()

    result = aggregated.join(initialTimestamps)
    result.to_csv('sequenceDb_'+str(k)+'.csv',index=False)

createSeqDb_filter_top_k(10000)
createSeqDb_filter_top_k(5000)
createSeqDb_filter_top_k(1000)
createSeqDb_filter_top_k(200)
createSeqDb_filter_top_k(10)
