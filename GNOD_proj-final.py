#!/usr/bin/env python
# coding: utf-8

# In[29]:


#imports
import pandas as pd
import numpy as np
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

#read files
top100 = pd.read_csv('top100songs.csv')
playlist_clustered = pd.read_csv('playlist_clustered.csv')
kmeans = pickle.load(open('kmeans_6.pkl', 'rb'))
scaler = pickle.load(open('scaler_spoti.pkl', 'rb'))

#clean files
top100['title'] = list(map(lambda x: x.lower(), top100['title']))
top100 = top100.drop('Unnamed: 0', axis=1)
playlist_clustered = playlist_clustered.drop('Unnamed: 0',axis=1)

#connect to API
secrets_file = open("SpotifySecret.txt","r")
string = secrets_file.read()
string.split('\n')
secrets_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        secrets_dict[line.split(':')[0]]=line.split(':')[1]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=secrets_dict['cid'],
                                                           client_secret=secrets_dict['cs']))

#get user input
song = input("Enter the name of a song: ")
song = song.lower()

#check if song in Top 100
top100_title = pd.Series(top100.title)
check = song in top100_title.unique()
if check == True:
#recommend another song from Top 100
    chosen_idx = np.random.choice(100, replace = False, size = 1)
    reco_100 = top100.iloc[chosen_idx]
    print("Your recommendation is:", reco_100['title'])
else:
#look for audio features of song
    results = sp.search(q='track:' + song, type = 'track')
    output = pd.DataFrame(results['tracks']['items'])
    output['name'] = output['name'].apply(lambda x: x.lower())

    if len(output) > 0:
        id = output['id'][0]
        track_info = sp.track(id)
        features_info = sp.audio_features(id)
    
        name = track_info['name']
        album = track_info['album']['name']
        artist = track_info['album']['artists'][0]['name']
        release_date = track_info['album']['release_date']
        length = track_info['duration_ms']
        popularity = track_info['popularity']
    
        acousticness = features_info[0]['acousticness']
        danceability = features_info[0]['danceability']
        energy = features_info[0]['energy']
        instrumentalness = features_info[0]['instrumentalness']
        liveness = features_info[0]['liveness']
        loudness = features_info[0]['loudness']
        speechiness = features_info[0]['speechiness']
        tempo = features_info[0]['tempo']
        time_signature = features_info[0]['time_signature']
    
        track_data = [name, album, artist, release_date, length, popularity, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]

        track_list = pd.DataFrame(track_data)
        total_features = track_list.T
        total_features.columns=['name', 'album', 'artist', 'release_date', 'length', 'popularity', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature']

#prepare the audio features for clustering
        X_list = total_features.drop(['name', 'album','artist','release_date'],axis=1)
        X_prep = scaler.transform(X_list)
        X_prep_def = pd.DataFrame(X_prep,columns=X_list.columns)

#assign song to a cluster
        prediction = kmeans.predict(X_prep_def)
        total_features['cluster_number'] = pd.Series(prediction, index=total_features.index)
    
#recommend another song from the same cluster
        reco_cluster = playlist_clustered.loc[playlist_clustered['cluster_number'] == prediction[0]][['name','artist','cluster_number']].reset_index(drop=True)
    
        chosen_idx_cluster = np.random.choice(len(reco_cluster), replace = False, size = 1)
        final_reco = reco_cluster.iloc[chosen_idx_cluster]
        reco_title = final_reco['name']
        reco_artist = final_reco['artist']
        print("Your recommendation is " + reco_title + ' by ' + reco_artist)

#if the song isn't on spotify at all
    else:
        print("Your song is unknown, please enter another song")

