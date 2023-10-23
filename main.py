from models.attributes import get_recommendations_attributes
from models.mfcc import get_recommendations_mfcc
from models.lyrics import get_recommendations_lyrics
import pandas as pd

def merged_model(song_title, song_artist, songs=pd.read_csv('data/more data.csv'), number_of_songs=2):

    dataset_attributes = get_recommendations_attributes(song_title, song_artist, songs, number_of_songs*10)

    dataset_att_mfcc = get_recommendations_mfcc(song_title, song_artist, dataset_attributes, number_of_songs*5)

    dataset_att_mfcc_lyrics = get_recommendations_lyrics(song_title, song_artist, dataset_att_mfcc, number_of_songs)

    return dataset_att_mfcc_lyrics