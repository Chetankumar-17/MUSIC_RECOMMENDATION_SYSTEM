import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Spotify API credentials
CLIENT_ID = "a995a98904e04c04a25ddf7593fcb881"
CLIENT_SECRET = "0ffef91ef9944749adb692fc62735209"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the music data from songs.csv
music = pd.read_csv('songs.csv')

# Load precomputed similarity data
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Function to get song album cover URL and Spotify URI
def get_song_album_cover_url_and_uri(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        uri = track["uri"]
        return album_cover_url, uri
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png", None  # Default image if not found

# Function to recommend similar songs based on selected song
# Function to recommend similar songs based on selected song
# Function to recommend similar songs based on selected song and filter by language
def recommend(selected_song):
    # Find the index of the selected song
    index = music[music['Song_name'] == selected_song].index[0]
    
    # Get the language of the selected song
    selected_language = music.iloc[index]['Language']
    
    # Calculate the distances of all songs to the selected song
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Prepare lists to store recommendations
    recommended_music_names = [music.iloc[index]['Song_name']]  # Add the selected song first
    recommended_music_posters = [get_song_album_cover_url_and_uri(music.iloc[index]['Song_name'], music.iloc[index]['Singer'])[0]]  # Add album cover of the selected song
    recommended_music_uris = [get_song_album_cover_url_and_uri(music.iloc[index]['Song_name'], music.iloc[index]['Singer'])[1]]  # Add URI of the selected song
    recommended_music_singers = [music.iloc[index]['Singer']]  # Add singer of the selected song

    # Add similar songs to the recommendation list, but only from the same language
    for i in distances[1:6]:  # Start from index 1 to skip the selected song itself
        if music.iloc[i[0]]['Language'] == selected_language:  # Check if the song has the same language
            artist = music.iloc[i[0]].Singer
            poster, uri = get_song_album_cover_url_and_uri(music.iloc[i[0]].Song_name, artist)
            recommended_music_posters.append(poster)
            recommended_music_names.append(music.iloc[i[0]].Song_name)
            recommended_music_uris.append(uri)
            recommended_music_singers.append(music.iloc[i[0]].Singer)

    return recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers


# Function to display songs by selected artist
def display_songs_by_artist(artist_name):
    artist_songs = music[music['Singer'] == artist_name]['Song_name'].tolist()
    artist_songs_posters_and_uris = [get_song_album_cover_url_and_uri(song, artist_name) for song in artist_songs]
    artist_songs_posters = [item[0] for item in artist_songs_posters_and_uris]
    artist_songs_uris = [item[1] for item in artist_songs_posters_and_uris]
    return artist_songs, artist_songs_posters, artist_songs_uris

# Function to recommend songs based on selected genre
# Function to recommend songs based on selected genre
# Function to recommend songs based on selected genre
def recommend_by_genre(genre):
    """
    Recommend songs based on the selected genre or all genres if "All" is selected.
    """
    # Normalize the input genre
    genre = genre.strip().lower()

    # Normalize the columns for consistent matching
    music['genre'] = music['genre'].str.strip().str.lower()
    music['Language'] = music['Language'].str.strip().str.lower()

    if genre == "all":
        # If "All" is selected, choose from all songs
        genre_songs = music['Song_name'].tolist()
    elif genre in ["hindi", "kannada", "telugu"]:  # Match for language-specific genres
        genre_songs = music[music['Language'] == genre]['Song_name'].tolist()
    else:
        # General genre matching
        genre_songs = music[music['genre'] == genre]['Song_name'].tolist()

    if not genre_songs:
        return [], [], [], []  # Return empty lists if no songs found for the genre

    # Ensure at least 4-5 recommendations by selecting random songs if necessary
    num_recommendations = 5
    if len(genre_songs) < num_recommendations:
        genre_songs = genre_songs + list(np.random.choice(music['Song_name'].tolist(), num_recommendations - len(genre_songs), replace=False))

    # Select a random song from the filtered list
    selected_song = np.random.choice(genre_songs)

    # Find the index of the selected song and calculate similarity
    index = music[music['Song_name'] == selected_song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Prepare recommendations
    recommended_music_names = []
    recommended_music_posters = []
    recommended_music_uris = []
    recommended_music_singers = []

    for i in distances[:num_recommendations]:  # Select up to 5 recommendations
        artist = music.iloc[i[0]].Singer
        poster, uri = get_song_album_cover_url_and_uri(music.iloc[i[0]].Song_name, artist)
        recommended_music_posters.append(poster)
        recommended_music_names.append(music.iloc[i[0]].Song_name)
        recommended_music_uris.append(uri)
        recommended_music_singers.append(music.iloc[i[0]].Singer) 

    return recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers



# Function to plot music statistics
def plot_music_statistics():
    # Artist with the most songs
    artist_song_count = music['Singer'].value_counts().head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=artist_song_count.values, y=artist_song_count.index, ax=ax1)
    ax1.set_title('Top 10 Artists with the Most Songs', fontsize=16, fontweight='bold', color='blue')
    ax1.set_xlabel('Number of Songs', fontsize=14, fontweight='bold', color='green')
    ax1.set_ylabel('Artist', fontsize=14, fontweight='bold', color='red')

    # Number of songs by genre
    genre_song_count = music['genre'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=genre_song_count.values, y=genre_song_count.index, ax=ax2)
    ax2.set_title('Top 10 Genres with the Most Songs', fontsize=16, fontweight='bold', color='blue')
    ax2.set_xlabel('Number of Songs', fontsize=14, fontweight='bold', color='green')
    ax2.set_ylabel('Genre', fontsize=14, fontweight='bold', color='red')

    return fig1, fig2

# Streamlit UI
st.sidebar.header('Music Recommender System')

# Selectbox to choose between different options
option = st.sidebar.selectbox("Choose an option", ["Recommend Songs", "Display Songs by Artist", "Recommend Songs by Genre"])

if option == "Recommend Songs":
    music_list = music['Song_name'].values
    selected_song = st.selectbox("Select a song from the dropdown", music_list)

    if st.button('Show Recommendation'):
        recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers = recommend(selected_song)
        cols = st.columns(5)  # Create 5 columns for the recommendations
        for col, name, poster, uri, singer in zip(cols, recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers):
            with col:
                st.image(poster, width=100)
                st.markdown(f'<p onclick="playSong(\'{uri}\')">{name} by {singer}</p>', unsafe_allow_html=True)

elif option == "Display Songs by Artist":
    all_artists = music['Singer'].unique()
    selected_artist = st.selectbox("Select an artist", all_artists)

    if st.button('Show Songs'):
        songs_by_artist, posters_by_artist, uris_by_artist = display_songs_by_artist(selected_artist)
        cols = st.columns(5)  # Create 5 columns for the songs
        for col, song, poster, uri in zip(cols, songs_by_artist, posters_by_artist, uris_by_artist):
            with col:
                st.image(poster, width=200)
                st.markdown(f'<p onclick="playSong(\'{uri}\')">{song}</p>', unsafe_allow_html=True)

elif option == "Recommend Songs by Genre":
    all_genres = music['genre'].unique()
    selected_genre = st.selectbox("Select a genre", all_genres)

    if st.button('Show Recommendation'):
        recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers = recommend_by_genre(selected_genre)
        cols = st.columns(5)  # Create 5 columns for the recommendations
        for col, name, poster, uri, singer in zip(cols, recommended_music_names, recommended_music_posters, recommended_music_uris, recommended_music_singers):
            with col:
                st.image(poster, width=200)
                st.markdown(f'<p onclick="playSong(\'{uri}\')">{name} by {singer}</p>', unsafe_allow_html=True)

# Music Statistics
st.sidebar.markdown("---")
st.sidebar.subheader("Music Statistics")

if st.sidebar.button("Show Music Statistics"):
    fig1, fig2 = plot_music_statistics()
    st.pyplot(fig1)
    st.pyplot(fig2)
