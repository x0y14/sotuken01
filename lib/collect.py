import spotipy

from lib.db import Track
from lib.spam import include_spam_text, is_include_spam_artist

MARKET = "JP"


def search_albums(sc: spotipy.client.Spotify, q: str, limit: int, offset: int):
    gain_albums = []

    try:
        result = sc.search(
            q=q,
            limit=limit,
            offset=offset,
            type="album",
            market=MARKET,
        )
    except spotipy.SpotifyException:
        raise

    for album in result["albums"]["items"]:
        # アルバム以外だったらスルー
        if album["album_type"] != "album":
            continue
        # スパムっぽかったらスルー
        if is_include_spam_artist(album["artists"]) or include_spam_text(album["name"]):
            continue
        # 取得したアルバムとして保存
        gain_albums.append(album)

    return gain_albums


def get_album_tracks(sc: spotipy.client.Spotify, album_id: str, limit: int, offset: int):
    gain_tracks = []
    try:
        result = sc.album_tracks(
            album_id=album_id,
            limit=limit,
            offset=offset,
            market=MARKET,
        )
    except spotipy.SpotifyException:
        raise

    for track in result["items"]:
        gain_tracks.append(track)

    return gain_tracks


def get_track_features(
        sc: spotipy.client.Spotify,
        album_id: str,
        track_id: str,
        track_name: str) -> Track:
    features = sc.audio_features(track_id)[0]
    if features is None:
        features = {}
    return Track(
        id=track_id,
        name=track_name,
        album_id=album_id,
        acousticness=features["acousticness"] if "acousticness" in features else 0,
        danceability=features["danceability"] if "danceability" in features else 0,
        duration_ms=features["duration_ms"] if "duration_ms" in features else 0,
        energy=features["energy"] if "energy" in features else 0,
        instrumentalness=features["instrumentalness"] if "instrumentalness" in features else 0,
        key=features["key"] if "key" in features else 0,
        liveness=features["liveness"] if "liveness" in features else 0,
        loudness=features["loudness"] if "loudness" in features else 0,
        mode=features["mode"] if "mode" in features else 0,
        speechiness=features["speechiness"] if "speechiness" in features else 0,
        tempo=features["tempo"] if "tempo" in features else 0,
        time_signature=features["time_signature"] if "time_signature" in features else 0,
        valence=features["valence"] if "valence" in features else 0,
    )
