import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify Credentials
CLIENT_ID = "ee9112c0450740d7b7a4419eda35b343"
CLIENT_SECRET = "93619cbb9ba2462b8a4fa3e6515c0095"

# Initialize Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to fetch album cover URL from Spotify
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")
    
    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return track["album"]["images"][0]["url"]
    return "https://i.postimg.cc/0QNxYz4V/social.png"  # Default image if not found

# Tokenization function (Splitting text into tokens)
def tokenize_text(text):
    return text.split(" ")

# Recommendation function based on cosine similarity
def recommend(song, music_df, similarity_matrix):
    try:
        index = music_df[music_df['song'] == song].index[0]
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])

        recommended_music_names = []
        recommended_music_posters = []

        for i in distances[1:6]:  # Top 5 recommendations
            song_name = music_df.iloc[i[0]].song
            artist_name = music_df.iloc[i[0]].artist
            recommended_music_names.append(song_name)
            recommended_music_posters.append(get_song_album_cover_url(song_name, artist_name))

        return recommended_music_names, recommended_music_posters
    except IndexError:
        return [], []

# Streamlit app UI
st.title("Music Recommendation System")

# Load dataset
try:
    music_df = pd.read_csv("C:/Users/chetan/OneDrive/Desktop/Music/spotify_millsongdata.csv")  # Replace with your Kaggle dataset path
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Check if 'genre' column exists, if not use only 'song' and 'artist' columns
if 'genre' in music_df.columns:
    st.text("Generating similarity matrix with 'song', 'artist', and 'genre'...")
    music_df['combined'] = music_df['song'] + " " + music_df['artist'] + " " + music_df['genre']
else:
    st.text("Generating similarity matrix with 'song' and 'artist'...")
    music_df['combined'] = music_df['song'] + " " + music_df['artist']

# Tokenization and Vectorization using TF-IDF
tokenizer = TfidfVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize_text)
tfidf_matrix = tokenizer.fit_transform(music_df['combined'])

# Apply TruncatedSVD for dimensionality reduction
n_components = 100  # You can adjust this number based on your system's capacity
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_svd = svd.fit_transform(tfidf_matrix)

# Compute the cosine similarity on the reduced matrix
similarity_matrix = cosine_similarity(tfidf_matrix_svd)
st.text("Similarity matrix generated.")

# Search options
st.subheader("Search for Songs")
search_by = st.radio("Search by", ["Song Name", "Artist", "Genre"])

# Filter dataset based on search input
if search_by == "Song Name":
    search_input = st.text_input("Enter Song Name").lower()
    filtered_df = music_df[music_df['song'].str.contains(search_input, case=False)] if search_input else music_df
elif search_by == "Artist":
    search_input = st.text_input("Enter Artist Name").lower()
    filtered_df = music_df[music_df['artist'].str.contains(search_input, case=False)] if search_input else music_df
else:  # Genre
    search_input = st.text_input("Enter Genre").lower()
    filtered_df = music_df[music_df['genre'].str.contains(search_input, case=False)] if search_input and 'genre' in music_df.columns else music_df

# Show filtered results
if search_input:
    st.write(f"Found {len(filtered_df)} songs matching your search.")
else:
    st.write("Showing all songs.")

# Dropdown to select a song
music_list = filtered_df['song'].values
selected_song = st.selectbox("Select a song to get recommendations", music_list)

# Show recommendations
if st.button('Get Recommendations'):
    recommended_music_names, recommended_music_posters = recommend(selected_song, music_df, similarity_matrix)
    
    if not recommended_music_names:
        st.error("No recommendations available. Please try another song.")
    else:
        # Display recommendations
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(recommended_music_names):
                with col:
                    st.text(recommended_music_names[i])
                    st.image(recommended_music_posters[i])
