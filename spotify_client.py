import requests
import urllib.parse

from datetime import datetime, timedelta
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
        if not self.is_token_valid():
            self.refresh_token()
        
        if self.device_id is None:
            self.get_device_id()
        
        print()
        print("query: " + query)
        print()
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
            return "Successfully started playing the requested song"
        else:
            return f"Failed to play song. Status code: {response.status_code}"

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
