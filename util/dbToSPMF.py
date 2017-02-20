import pandas as pd

filePath = 'C:/Users/Umberto/Dropbox/datasets/music/sessions.csv'
file = pd.read_csv(filePath)

groups = file.groupby('session_id')
#songs as strings

file['song_id'] = file['song_id'].astype(str)
agg=groups['song_id'].apply(lambda x: "%s" % ', '.join(x))

with open('sequences.txt','w') as fout:
    for i,row in pd.DataFrame(agg).iterrows():
        fout.write(row['song_id'].replace(", ", " -1 "))
        fout.write(' -2\n')


