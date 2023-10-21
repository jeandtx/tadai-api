import pandas as pd
import pickle
dataset = pd.read_csv('./data/more data.csv')
import re
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
import string
import re
import lyricsgenius

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def to_lower(text):
    return text.lower()

def tokenize(text):
    return word_tokenize(text)
english_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in english_stopwords]

def preprocess(text):
    text = remove_punctuation(text)
    text = to_lower(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return tokens

dataset['tokenized_lyrics'] = dataset['Lyrics'].apply(preprocess)

model = pickle.load(open("./models/model.pkl", "rb"))

def get_song_vector(tokenized_lyrics):
    vectors = [model.wv[word] for word in tokenized_lyrics if word in model.wv]
    return sum(vectors) / len(vectors) if vectors else []

dataset['song_vector'] = dataset['tokenized_lyrics'].apply(get_song_vector)
API_KEY = "Vwb4Vm4moXv_6iYeDa_ibcfx3htmd0osTnOGDRaXmzwAmSJ6IfLu5POZ-_uab3sx"

def fetch_lyrics(artist_name, song_title):
    genius = lyricsgenius.Genius(API_KEY)
    song = genius.search_song(song_title, artist_name)
    return song.lyrics if song else None

def clean_and_preprocess(lyrics):
    # Nettoyage
    lyrics = lyrics.split('Lyrics', 1)[-1].rsplit('Embed', 1)[0]
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics = re.sub(r'\(.*?\)', '', lyrics)
    lyrics = ' '.join(lyrics.split())
    
    # Prétraitement
    lyrics = remove_punctuation(lyrics)
    lyrics = to_lower(lyrics)
    tokenized_lyrics = tokenize(lyrics)
    tokenized_lyrics = remove_stopwords(tokenized_lyrics)
    
    # Converting list of tokens back to a single string
    cleaned_and_processed_lyrics = ' '.join(tokenized_lyrics)
    
    return cleaned_and_processed_lyrics

def get_vector_for_lyrics(lyrics):
    tokenized_lyrics = lyrics.split()
    return get_song_vector(tokenized_lyrics)

def get_top_keywords_for_vector(vector, top_n=5):
    # Calculate cosine similarity between the input vector and all word vectors in the model
    similarities = {word: cosine_similarity([vector], [model.wv[word]])[0][0] for word in model.wv.index_to_key}
    
    # Sort words by similarity
    sorted_similar_words = sorted(similarities.keys(), key=lambda word: similarities[word], reverse=True)
    
    return sorted_similar_words[:top_n]

def recommend_top_10_songs_and_artists_with_keywords(artist_name, song_title):
    lyrics = fetch_lyrics(artist_name, song_title)
    if not lyrics:
        return "Could not fetch the lyrics. Please try another song."
    
    cleaned_lyrics = clean_and_preprocess(lyrics)
    print(cleaned_lyrics)
    input_vector = get_vector_for_lyrics(cleaned_lyrics)
    
    user_song_keywords = get_top_keywords_for_vector(input_vector)
    print(f"Keywords for user's song ({song_title} by {artist_name}): {', '.join(user_song_keywords)}\n")
    
    similarities = [cosine_similarity([input_vector], [song_vector])[0][0] for song_vector in dataset['song_vector']]
    top_10_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:10]
    
    recommendations = []
    for idx in top_10_indices:
        song_name = dataset['Name'].iloc[idx]
        song_artist = dataset['Artist'].iloc[idx]
        song_vector = dataset['song_vector'].iloc[idx]
        song_keywords = get_top_keywords_for_vector(song_vector)
        recommendations.append(f"{song_name} by {song_artist} (Keywords: {', '.join(song_keywords)})")
    
    return recommendations

artist = input("Enter an artist: ")
song = input("Enter a song: ")
recommended_top_10_songs_and_artists = recommend_top_10_songs_and_artists_with_keywords(artist, song)
print(recommended_top_10_songs_and_artists)