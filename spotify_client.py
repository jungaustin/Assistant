import requests
import urllib.parse

from datetime import datetime, timedelta
from rapidfuzz import process
# from flask import Flask, redirect, request, jsonify, session

import os
from dotenv import load_dotenv
load_dotenv()

class SpotifyClient:
    def __init__(self):
        self.CLIENT_ID = os.getenv("CLIENT_ID")
        self.CLIENT_SECRET = os.getenv("CLIENT_SECRET")
        self.REFRESH_TOKEN =  os.getenv("REFRESH_TOKEN")
        self.access_token = None
        self.device_id = None
        self.expires_at = None
        self.API_BASE_URL = 'https://api.spotify.com/v1'
        self.TOKEN_URL = 'https://accounts.spotify.com/api/token'
        self.playlists = None
    
    def is_token_valid(self):
        return self.access_token is not None and datetime.now().timestamp() < self.expires_at
    
    def get_headers(self):
        return {
            'Authorization': f"Bearer {self.access_token}"
        }
    
    def refresh_token(self):
        if not self.is_token_valid():
            req_body = {
                'grant_type' : 'refresh_token',
                'refresh_token' : self.REFRESH_TOKEN,
                'client_id' : self.CLIENT_ID,
                'client_secret' : self.CLIENT_SECRET
            }
            response = requests.post(self.TOKEN_URL, data = req_body)
            new_token_info = response.json()
            self.access_token = new_token_info['access_token']
            self.expires_at = datetime.now().timestamp() + new_token_info['expires_in']

    def get_device_id(self):
        if not self.is_token_valid():
            self.refresh_token()
        
        response = requests.get(self.API_BASE_URL + '/me/player/devices', headers=self.get_headers())
        devices = response.json()
        for device in devices['devices']:
            #I can change this to whichever device I want later (currently using mac)
            if 'MacBook' in device['name']:
                self.device_id = device['id']
                break

    def play_song(self, query : str) -> str:
        print(query)
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()

        search = requests.get(
            self.API_BASE_URL + '/search',
            headers=self.get_headers(),
            params={
                'q': query,
                'type': 'track',
                'limit': 1
            }
        )
        search_json = search.json()
        items = search_json.get('tracks', {}).get('items', [])
        if items and 'uri' in items[0]:
            track_uri = items[0]['uri']
        else:
            return "No matching track found."

        data = {
            'uris' : [track_uri],
            'device_id' : self.device_id
        }
        response = requests.put(self.API_BASE_URL + f'/me/player/play',
                                headers = self.get_headers(),
                                json = data
                                )
        if response.status_code == 204:
            return "Playing."
        else:
            return f"Failed to play song. Status code: {response.status_code}"
    
    def get_my_playlists(self):
        if not self.is_token_valid():
            self.refresh_token()

        if self.device_id is None:
            self.get_device_id()

        offset = 0
        params={
            'limit': 50,
            'offset': offset,
        }
        
        my_playlists = {}
        
        while(True):
            search = requests.get(
                self.API_BASE_URL + '/me/playlists',
                headers=self.get_headers(),
                params=params
            )
            search_res = search.json()
            items = search_res.get('items', {})
            if search.status_code != 200:
                return (f"Error fetching playlists: {search.status_code} - {search.text}")
            for item in items:
                print(item)
                my_playlists[item['name']] = item['uri']
            if(len(items) < 50):
                break
            offset += 50
        self.playlists = my_playlists
        return self.playlists.keys

    def get_best_playlist_match(self, user_input: str, threshold=70) -> str | None:
        print("h2")
        result = process.extractOne(user_input, self.playlists.keys(), score_cutoff=threshold)
        print(result[0] if result else None)
        return result[0] if result else None
    
    def play_playlist(self, input : str) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()

        if(self.playlists == None):
            _ = self.get_my_playlists()
        
        for name in self.playlists.keys():
            if input.strip().lower() == name.strip().lower():
                playlist_name = name
                break

        if playlist_name is None:
            playlist_name = self.get_best_playlist_match(input)

        if playlist_name is None:
            return f"No matching playlist found for '{input}'."
        
        playlist_uri = self.playlists[playlist_name]
        data = {
            'context_uri': playlist_uri,
            'position_ms': 0,
        }
        response = requests.put(
            self.API_BASE_URL + '/me/player/play',
            params={
                'device_id': self.device_id
                },
            headers=self.get_headers(),
            json=data
        )
        if response.status_code == 204:
            return f"Playing '{playlist_name}'."
        else:
            print(response)
            return f"Failed to play playlist. Status code: {response.status_code}"
    
    def shuffle(self, state : bool) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
            
        response = requests.put(
            self.API_BASE_URL + '/me/player/shuffle', 
            headers=self.get_headers(),
            params={'state': state, 'device_id': self.device_id}
        )
        if response.status_code > 199 and response.status_code < 300:
            return f"It worked."
        else:
            print(response)
            return f"Failed to shuffle"
    
    def pause_playback(self) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
        
        response = requests.put(
            self.API_BASE_URL + '/me/player/pause',
            headers=self.get_headers(),
            params={
                'device_id': self.device_id
            }
        )
        
        if response.status_code > 199 or response.status_code < 300:
            return "Paused"
        else:
            return f"An Error has Occured: Status Code {response.status_code}"
        
    def play_playback(self) -> str:
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
        
        response = requests.put(
            self.API_BASE_URL + '/me/player/play',
            headers=self.get_headers(),
            params={
                'device_id': self.device_id
            }
        )
        
        if response.status_code > 199 or response.status_code < 300:
            return "Started playback"
        else:
            return f"An error has occured: Status Code {response.status_code}"


# def queue_song():
#     if 'access_token' not in session:
#         return redirect('/login')
    
#     if datetime.now().timestamp() > session['expires_at']:
#         return redirect('/refresh-token?next=/queue-song')
    
#     if 'device_id' not in session or session['device_id'] == None:
#         return redirect('/device?next=/queue-song')
    
#     headers = {
#         'Authorization': f"Bearer {session['access_token']}"
#     }
#     #change this later with song from voice to text, maybe use a llm to give json of this type
#     search = requests.get(
#         API_BASE_URL + '/search',
#         headers=headers,
#         params={
#             'q': 'track:Ferris Wheel artist:QWER',
#             'type': 'track',
#             'limit': 1
#         }
#     )
#     search_json = search.json()
#     track_uri = search_json['tracks']['items'][0]['uri']
#     response = requests.post(API_BASE_URL + f'/me/player/queue',
#                              headers = headers,
#                              params = {
#                                  'device_id' : session['device_id'],
#                                  'uri' : track_uri
#                                  }
#                              )
#     return redirect('/')
