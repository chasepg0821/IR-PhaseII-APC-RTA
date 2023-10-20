import os
import tqdm
import json
import numpy as np
import tensorflow as tf


class GlobalData():
    def __init__(self):
        self.track_count = 0
        self.tracks = {}
        self.artists = {}
        self.playlists = {}
        self.albums = {}

    def add_track(self, uri, track):
        entry = self.tracks.get(uri)
        if entry:
            entry.add_count()
        else:
            self.tracks[uri] = track

    def add_artist(self, uri, name):
        entry = self.artists.get(uri)
        if not entry:
            self.artists[uri] = {"name": name, "id": len(self.artists) / 1.0}

    def add_playlist(self, p_id, playlist):
        entry = self.playlists.get(p_id)
        if not entry:
            self.playlists[p_id] = playlist

    def add_album(self, uri, name):
        entry = self.albums.get(uri)
        if not entry:
            self.albums[uri] = {"name": name, "id": len(self.albums) / 1.0}


class Track():
    def __init__(self, track_info):
        self.pop = 1.0
        self.dur = track_info['duration_ms'] / 1000.0
        self.artist = track_info['artist_uri'].lstrip('spotify:artist:')
        self.album = track_info['album_uri'].lstrip('spotify:album:')

    def add_count(self):
        self.pop += 1.0

    def get_vector_rep(self, gd):
        return np.array([self.pop, self.dur, gd.artists[self.artist]['id'], gd.albums[self.album]['id']])


class Playlist():
    def __init__(self, track_ids, gd):
        self.num_tracks = len(track_ids)
        self.tracks = track_ids
        tensor = []
        for uri in track_ids:
            tensor.append(gd.tracks[uri].get_vector_rep(gd))
        tensor = np.array(tensor)
        self.vector_rep = tf.reduce_sum(tensor, 0).numpy()


def format_data(gd):
    filenames = os.listdir(os.getcwd())
    for filename in tqdm.tqdm(sorted(filenames, key=str)):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((os.getcwd(), filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice["playlists"]:
                track_ids = []
                for track in playlist['tracks']:
                    trackObj = Track(track)
                    t_uri = track['track_uri'].lstrip('spotify:track:')
                    gd.add_track(t_uri, trackObj)
                    gd.add_artist(trackObj.artist, track['artist_name'])
                    gd.add_album(trackObj.album, track['album_name'])
                    track_ids.append(t_uri)
                gd.add_playlist(playlist['pid'], Playlist(track_ids, gd))


if __name__ == "__main__":
    gd = GlobalData()
    format_data(gd)

    features = []
    labels = []
    for key in gd.playlists.keys():
        for track in gd.playlists[key].tracks:
            # if len(features) > 0:
            #     features = np.append(features, gd.playlists[key].vector_rep)
            # else:
            #     features = gd.playlists[key].vector_rep
            features.append(gd.playlists[key].vector_rep)
            labels.append(gd.tracks[track].get_vector_rep(gd))
    features = np.array(features)
    labels = np.array(labels)

    print(features)
    print(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(4)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam())

    model.fit(features, labels, epochs=10)
