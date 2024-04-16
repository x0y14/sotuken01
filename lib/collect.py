import spotipy

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
