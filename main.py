from models.attributes import get_recommendations_attributes
from models.mfcc import get_recommendations_mfcc
from models.lyrics import get_recommendations_lyrics
import pandas as pd
import time

def merged_model(song_title, song_artist, songs=pd.read_csv('data/more data.csv'), number_of_songs=2):
    start_time = time.time()
    dataset_attributes = get_recommendations_attributes(song_title, song_artist, songs, number_of_songs*10)
    step1_time = time.time()
    print(f"Step 1 (Attributes) Execution Time: {step1_time - start_time:.2f} seconds")

    dataset_att_mfcc = get_recommendations_mfcc(song_title, song_artist, dataset_attributes, number_of_songs*5)
    step2_time = time.time()
    print(f"Step 2 (MFCC) Execution Time: {step2_time - step1_time:.2f} seconds")

    dataset_att_mfcc_lyrics = get_recommendations_lyrics(song_title, song_artist, dataset_att_mfcc, number_of_songs)

    end_time = time.time()
    print(f"Step 3 (Lyrics) Execution Time: {end_time - step2_time:.2f} seconds")
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

    return dataset_att_mfcc_lyrics
