import os
import time

from spotipy.oauth2 import SpotifyClientCredentials

from lib.collect import *
from lib.db import *

SPOTIFY_SEARCH_LIMIT = 1000
# SPOTIFY_SEARCH_LIMIT = 100  # DEBUG
DB_NAME = "soundtrack.sqlite"


def main():
    # DBの準備
    conn = connect_db(DB_NAME)
    create_albums_table_if_not_exists(conn)
    create_tracks_table_if_not_exists(conn)
    create_genres_table_if_not_exists(conn)

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
            if is_album_exists(conn, album['id']):  # DBに有るのでスキップ
                continue
            insert_album(conn, Album(id=album['id'], name=album['name']))
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
                if is_track_exists(conn, track['id']):
                    continue
                track_data = get_track_features(
                    sc=sc,
                    album_id=album['id'],
                    track_id=track['id'],
                    track_name=track['name'])
                insert_track(conn, track_data)
                print(f"    * {track['name']}: {track['id']}")
                time.sleep(1)

    conn.close()


if __name__ == "__main__":
    main()
