import requests
import urllib.parse

from datetime import datetime, timedelta
from flask import Flask, redirect, request, jsonify, session

import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

app.secret_key = 'RANDOM STRING'

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = 'http://127.0.0.1:5000/callback'

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1'

@app.route('/')
def index():
    return "Welcome to my Spotify App <a href='/login'>Login with Spotify</a>"

@app.route('/login')
def login():
    scope = (
        'user-read-playback-state '
        'user-modify-playback-state '
        'user-read-currently-playing '
        'playlist-read-private '
        'playlist-read-collaborative'
    )
    
    params = {
        'client_id' : CLIENT_ID,
        'response_type' : 'code',
        'scope' : scope,
        'redirect_uri' : REDIRECT_URI,
        'show_dialog' : True
    }
    
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    3
    return redirect(auth_url)

@app.route('/callback')
def callback():
    if 'error' in request.args:
        return jsonify({"error": request.args['error']})
    
    if 'code' in request.args:
        req_body = {
            'code' : request.args['code'],
            'grant_type' : 'authorization_code',
            'redirect_uri' : REDIRECT_URI,
            'client_id' : CLIENT_ID,
            'client_secret' : CLIENT_SECRET
        }
        
        response = requests.post(TOKEN_URL, data= req_body)
        token_info = response.json()
        print(token_info)
        
        session['access_token'] = token_info['access_token']
        session['refresh_token'] = token_info['refresh_token']
        session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']
        
        return redirect('play-song')

@app.route('/playlists')
def get_playlists():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')
    
    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }
    
    response = requests.get(API_BASE_URL + 'me/playlists', headers = headers)
    playlists = response.json()
    
    return jsonify(playlists)

@app.route('/play-song')
def play_song():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token?next=/play-song')
    
    if 'device_id' not in session:
        return redirect('/device?next=/play-song')
    
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
    data = {
        'uris' : [track_uri],
        'device_id' : session['device_id']
    }
    response = requests.put(API_BASE_URL + f'/me/player/play',
                             headers = headers,
                             json = data
                             )
    return redirect('/')

@app.route('/queue-song')
def queue_song():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token?next=/queue-song')
    
    if 'device_id' not in session or session['device_id'] == None:
        print('here11111')
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

@app.route('/device')
def get_device_id():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')
    
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
    
    next_url = request.args.get('next', '/play-song')
    return redirect(next_url)


@app.route('/refresh-token')
def refresh_token():
    if 'refresh_token' not in session['refresh_token']:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
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
        
        next_url = request.args.get('next', '/')
        return redirect(next_url)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)