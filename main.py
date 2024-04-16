import os
from pprint import pprint

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from collect import search_albums, get_album_tracks

# SPOTIFY_SEARCH_LIMIT = 1000
SPOTIFY_SEARCH_LIMIT = 100  # DEBUG


def main():
    # Spotify APIの準備
    my_id = os.environ["SPOTIFY_CLIENT_ID"]
    my_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    ccm = SpotifyClientCredentials(client_id=my_id, client_secret=my_secret)
    sc = spotipy.Spotify(client_credentials_manager=ccm)

    gain_albums = []
    album_request_limit = 50

    gain_tracks = []
    track_request_limit = 50

    while (SPOTIFY_SEARCH_LIMIT - len(gain_albums)) >= album_request_limit:  # limitより小さいということは、取り終えてるってこと...?
        # アルバム検索!
        albums = search_albums(
            sc=sc,
            q="オリジナル サウンドトラック",
            limit=album_request_limit,
            offset=len(gain_albums),
        )
        gain_albums.extend(albums)

        for album in albums:
            # アルバムとそのアーティストを取り出す!
            print(f"{album['name']}: {album['id']}")
            print("  artists:", ", ".join(f"{artist['name']}: {artist['id']}" for artist in album["artists"]))

            # 曲を取り出す!
            tracks = get_album_tracks(
                sc=sc,
                album_id=album['id'],
                limit=track_request_limit,
                offset=0,
            )
            gain_tracks.extend(tracks)
            for track in tracks:
                print(f"    * {track['name']}: {track['id']}")


if __name__ == "__main__":
    main()
