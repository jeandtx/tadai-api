import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
cid = os.getenv('SPOTIFY_CLIENT_ID')
secret = os.getenv('SPOTIFY_CLIENT_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

def get_song_data(song_title, song_artist):
    data = {}
    search_query = f"track:{song_title} artist:{song_artist}"
    results = sp.search(q=search_query, limit=1, type='track')
    data['Name'] = song_title
    if results['tracks']['items'] == []:
        raise ValueError('Song not found in Spotify')
    else:
        song = results['tracks']['items'][0]
        data['Artist'] = song['artists'][0]['name']
        data['Album'] = song['album']['name']
        data['Popularity'] = song['popularity']
        data['Lyrics'] = None
        data['Lyrics length'] = None
        data['Spotify ID'] = song['id']
        audio_features = sp.audio_features(song['id'])[0]
        data['Spotify features'] = audio_features
        data['Danceability'] = audio_features['danceability']
        data['Energy'] = audio_features['energy']
        data['loudness'] = audio_features['loudness']
        data['Speechiness'] = audio_features['speechiness']
        data['Acousticness'] = audio_features['acousticness']
        data['Instrumentalness'] = audio_features['instrumentalness']
        data['Liveness'] = audio_features['liveness']
        data['Valence'] = audio_features['valence']
        data['Tempo'] = audio_features['tempo']
        data['Duration'] = audio_features['duration_ms']
        data['Audio'] = song['preview_url']

    return pd.DataFrame([data])

def add_song(song_title, song_artist, dataframe):
    if len(dataframe.columns) != 19:
        raise ValueError('Dataframe must have 19 columns. Check if spotify features are included.')
    else:
        song = get_song_data(song_title, song_artist)
        if len(song) < 1:
            raise ValueError('Song not found')
        elif song['Spotify ID'][0] in dataframe['Spotify ID'].values:
            return dataframe
        else:
            dataframe = pd.concat([dataframe, song], ignore_index=True)
            dataframe.drop_duplicates(subset=['Spotify ID'], inplace=True)
            dataframe.to_csv('./data/more data.csv', index=False)
            return dataframe

def get_closest_songs(song_index, pca_df, songs, number_of_songs=10):
    song_row = pca_df.iloc[song_index]
    other_rows = pca_df.drop(song_index)
    distances = other_rows.apply(lambda row: np.linalg.norm(row - song_row), axis=1)
    closest_indexes = distances.sort_values().index[:number_of_songs]
    closest_songs = songs.iloc[closest_indexes]
    return closest_songs

def get_song_index(song_name, song_artist, songs):
    song_name = song_name.lower()
    song_artist = song_artist.lower()
    filtered_songs = songs[(songs['Name'].str.lower() == song_name) & (songs['Artist'].str.lower() == song_artist)]
    if not filtered_songs.empty:
        return filtered_songs.index[0]
    else:
        return None