import pandas as pd
import requests
import librosa as lb
import io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.functions import get_closest_songs, get_song_index, add_song

def extract_features_from_mp3(mp3_link, max_seq_len=None):
    audio = requests.get(mp3_link)
    y, sr = lb.load(io.BytesIO(audio.content))
    mfcc = lb.feature.mfcc(y=y, sr=sr)
    
    # If max_seq_len is provided, pad or truncate the sequence
    if max_seq_len:
        if mfcc.shape[1] < max_seq_len:
            pad_width = max_seq_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > max_seq_len:
            mfcc = mfcc[:, :max_seq_len]
    
    covariance = np.cov(mfcc)
    mean = mfcc.mean(0)
    return (mfcc, covariance, mean)

def compute_pca(dataframe, max_seq_len=None):
    data = []
    for index, song in dataframe.iterrows():
        feature = extract_features_from_mp3(song['Audio'], max_seq_len=max_seq_len)
        data.append(feature)
    feature_data = np.array(data, dtype=object)
    flattened_data = [np.hstack((mfcc.flatten(), covariance.flatten(), mean.flatten())) for (mfcc, covariance, mean) in feature_data]
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(flattened_data)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(normalized_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    return pca_df

def get_recommendations_mfcc(song_name, song_artist, songs=None, number_of_songs=10):
    if songs is None:
        songs = pd.read_csv('./data/more data.csv')
    songs = songs.copy()

    songs.dropna(subset=['Audio'], inplace=True)
    song_index = get_song_index(song_name, song_artist, songs)
    if song_index is None:
        songs = add_song(song_name, song_artist, songs)
        if isinstance(songs, str):
            print('No song sample found on Spotify -> bypassing MFCC PCA')
            return songs
        song_index = get_song_index(song_name, song_artist, songs)
    pca_df = compute_pca(songs)
    closest_songs = get_closest_songs(song_index, pca_df, songs, number_of_songs)    
    return closest_songs
