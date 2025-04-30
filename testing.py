import subprocess

def play_music(song_name= None, artist =None, playlist=None):
    subprocess.Popen(["open", "-a", "Spotify"])

play_music()