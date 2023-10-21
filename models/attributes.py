import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.functions import add_song, get_closest_songs, get_song_index

def compute_pca(dataframe):
    data = []
    for index, song in dataframe.iterrows():
        feature = (song['Danceability'], song['Energy'], song['loudness'], song['Speechiness'], song['Acousticness'], song['Instrumentalness'], song['Liveness'], song['Valence'], song['Tempo'], song['Duration'], song['Popularity'])
        data.append(feature)
    feature_data = np.array(data, dtype=object)  
    flattened_data = feature_data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(flattened_data)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(normalized_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    return pca_df

def get_recommendations_attributes(song_name, song_artist, songs=pd.read_csv('./data/more data.csv'), number_of_songs=10):
    song_index = get_song_index(song_name, song_artist, songs)
    if song_index is None:
        songs = add_song(song_name, song_artist, songs)
        song_index = get_song_index(song_name, song_artist, songs)
    pca_df = compute_pca(songs)
    closest_songs = get_closest_songs(song_index, pca_df, songs, number_of_songs)
    return closest_songs
