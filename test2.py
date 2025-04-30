import requests
import urllib.parse

from datetime import datetime, timedelta
from flask import Flask, redirect, request, jsonify, session

import os
from dotenv import load_dotenv
load_dotenv()

session = {}

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
session['refresh_token'] = os.getenv("REFRESH_TOKEN")

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1'

def index():
    return "Welcome to my Spotify App <a href='/login'>Login with Spotify</a>"


def refresh_token():
    if 'refresh_token' not in session:
        return 'Needs Refresh Token'
    
    if 'expires_at' not in session or datetime.now().timestamp() > session['expires_at']:
        req_body = {
            'grant_type' : 'refresh_token',
            'refresh_token' : session['refresh_token'],
            'client_id' : CLIENT_ID,
            'client_secret' : CLIENT_SECRET
        }
        
        response = requests.post(TOKEN_URL, data = req_body)
        new_token_info = response.json()
        
        session['access_token'] = new_token_info['access_token']
        session['expires_at'] = datetime.now().timestamp() + new_token_info['expires_in']
    
    return

# def get_playlists():
#     if 'access_token' not in session:
#         return redirect('/login')
    
#     if datetime.now().timestamp() > session['expires_at']:
#         return redirect('/refresh-token')
    
#     headers = {
#         'Authorization': f"Bearer {session['access_token']}"
#     }
    
#     response = requests.get(API_BASE_URL + 'me/playlists', headers = headers)
#     playlists = response.json()
    
#     return jsonify(playlists)

def play_song(query : str):
    if 'access_token' not in session:
        refresh_token()
        
    if 'expires_at' not in session or datetime.now().timestamp() > session['expires_at']:
        refresh_token()
    
    if 'device_id' not in session:
        get_device_id()
    
    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }
    # print('Access Token is: ' + session['access_token'])
    #change this later with song from voice to text, maybe use a llm to give json of this type
    search = requests.get(
        API_BASE_URL + '/search',
        headers=headers,
        params={
            'q': query,
            'type': 'track',
            'limit': 1
        }
    )
    search_json = search.json()
    track_uri = search_json['tracks']['items'][0]['uri']
    data = {
        'uris' : [track_uri],
        'device_id' : session['device_id']
    }
    response = requests.put(API_BASE_URL + f'/me/player/play',
                             headers = headers,
                             json = data
                             )
    if response.status_code == 204:
        return "Final Answer: Successfully started playing the requested song"
    else:
        return f"Failed to play song. Status code: {response.status_code}"

def queue_song():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token?next=/queue-song')
    
    if 'device_id' not in session or session['device_id'] == None:
        return redirect('/device?next=/queue-song')
    
    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }
    #change this later with song from voice to text, maybe use a llm to give json of this type
    search = requests.get(
        API_BASE_URL + '/search',
        headers=headers,
        params={
            'q': 'track:Ferris Wheel artist:QWER',
            'type': 'track',
            'limit': 1
        }
    )
    search_json = search.json()
    track_uri = search_json['tracks']['items'][0]['uri']
    response = requests.post(API_BASE_URL + f'/me/player/queue',
                             headers = headers,
                             params = {
                                 'device_id' : session['device_id'],
                                 'uri' : track_uri
                                 }
                             )
    return redirect('/')

def get_device_id():
    if 'access_token' not in session:
        refresh_token()
    
    if 'expires_at' not in session or datetime.now().timestamp() > session['expires_at']:
        refresh_token()
    
    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }
    
    response = requests.get(API_BASE_URL + '/me/player/devices', headers=headers)
    devices = response.json()
    device_id = None
    for device in devices['devices']:
        if 'MacBook' in device['name']:
            device_id = device['id']
            break
    
    session['device_id'] = device_id
    
    return


# OG
# def refresh_token():
#     if 'refresh_token' not in session['refresh_token']:
#         return redirect('/login')
    
#     if datetime.now().timestamp() > session['expires_at']:
#         req_body = {
#             'grant_type' : 'refresh_token',
#             'refresh_token' : session['refresh_token'],
#             'client_id' : CLIENT_ID,
#             'client_secret' : CLIENT_SECRET
#         }
        
#         response = requests.post(TOKEN_URL, data = req_body)
#         new_token_info = response.json()
        
        
#         session['access_token'] = new_token_info['access_token']
#         session['expires_at'] = datetime.now().timestamp() + new_token_info['expires_in']
        
#         next_url = request.args.get('next', '/')
#         return redirect(next_url)
