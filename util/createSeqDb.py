import pandas as pd


def createSeqDbList():
    filePath = 'C:/Users/Umberto/Dropbox/datasets/music/sessions.csv'
    file = pd.read_csv(filePath)

    #song_id to string
    #file['song_id'] = file['song_id'].astype(str)

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


