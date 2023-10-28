from models.mfcc import get_recommendations_mfcc
from models.functions import get_song_index
import pandas as pd
from main import merged_model

songs = None
if songs is None:
    songs = pd.read_csv('./data/more data.csv')

songs = songs.copy()

songs.dropna(subset=['Audio'], inplace=True)



rec = merged_model('Beso', 'ROSALÍA', songs, 10)
print(rec) 