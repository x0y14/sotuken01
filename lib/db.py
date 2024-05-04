import sqlite3
from dataclasses import dataclass
from typing import List


@dataclass
class Track:
    id: str
    name: str
    album_id: str
    acousticness: float
    danceability: float
    duration_ms: int
    energy: float
    instrumentalness: float
    key: int
    liveness: float
    loudness: float
    mode: int
    speechiness: float
    tempo: float
    time_signature: int
    valence: float


@dataclass
class Album:
    id: str
    name: str


def connect_db(db_name: str) -> sqlite3.Connection:
    return sqlite3.connect(db_name)


def create_tracks_table_if_not_exists(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """create table if not exists tracks(
            id text primary key,
            name text,
            album_id text, 
            acousticness real,
            danceability real,
            duration_ms integer,
            energy real,
            instrumentalness real,
            "key" integer,
            liveness real,
            loudness real,
            mode integer,
            speechiness real,
            tempo real,
            time_signature integer,
            valence real);"""
    )
    conn.commit()


def create_albums_table_if_not_exists(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """create table if not exists albums( id text primary key, name text )"""
    )
    conn.commit()


def create_genres_table_if_not_exists(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute(
        """create table if not exists genres( id text primary key, genres text )"""
    )
    conn.commit()


def insert_track(conn: sqlite3.Connection, track: Track):
    cursor = conn.cursor()
    cursor.execute(
        """insert or replace into tracks(
        id, 
        name, 
        album_id, 
        acousticness, 
        danceability, 
        duration_ms, 
        energy, 
        instrumentalness, 
        key, 
        liveness, 
        loudness,
        mode, 
        speechiness,
        tempo,
        time_signature,
        valence) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);""",
        (track.id,
         track.name,
         track.album_id,
         track.acousticness,
         track.danceability,
         track.duration_ms,
         track.energy,
         track.instrumentalness,
         track.key,
         track.liveness,
         track.loudness,
         track.mode,
         track.speechiness,
         track.tempo,
         track.time_signature,
         track.valence)
    )
    conn.commit()


def insert_album(conn: sqlite3.Connection, album: Album):
    cursor = conn.cursor()
    cursor.execute(
        """insert or replace into albums(id, name) values (?, ?);""",
        (album.id, album.name)
    )
    conn.commit()


def is_track_exists(conn: sqlite3.Connection, track_id: str) -> bool:
    cursor = conn.cursor()
    cursor.execute("select * from tracks where id=?", (track_id,))
    return cursor.fetchone() is not None


def is_album_exists(conn: sqlite3.Connection, album_id: str) -> bool:
    cursor = conn.cursor()
    cursor.execute("select * from albums where id=?", (album_id,))
    return cursor.fetchone() is not None


def get_album_list(conn: sqlite3.Connection, limit: int, offset: int) -> List[Album]:
    cursor = conn.cursor()
    cursor.execute("""select * from albums limit ? offset ?;""", (limit, offset,))
    album_datas = cursor.fetchall()

    album_list = []
    album_cols = ["id", "name"]
    for album_data_only in album_datas:
        album_data = dict(zip(album_cols, album_data_only))
        album_list.append(Album(id=album_data["id"], name=album_data["name"]))

    return album_list


def get_tracks(conn: sqlite3.Connection, album_id: str, limit: int, offset: int) -> List[Track]:
    cursor = conn.cursor()
    cursor.execute(
        """select * from tracks where album_id=? limit ? offset ?;""",
        (album_id, limit, offset,)
    )
    track_datas = cursor.fetchall()

    track_list = []
    track_cols = [
        "id", "name", "album_id", "acousticness",
        "danceability", "duration_ms", "energy",
        "instrumentalness", "key", "liveness",
        "loudness", "mode", "speechiness",
        "tempo", "time_signature", "valence"]
    for track_data_only in track_datas:
        track_data = dict(zip(track_cols, track_data_only))
        track_list.append(
            Track(
                id=track_data["id"],
                name=track_data["name"],
                album_id=track_data["album_id"],
                acousticness=track_data["acousticness"],
                danceability=track_data["danceability"],
                duration_ms=track_data["duration_ms"],
                energy=track_data["energy"],
                instrumentalness=track_data["instrumentalness"],
                key=track_data["key"],
                liveness=track_data["liveness"],
                loudness=track_data["loudness"],
                mode=track_data["mode"],
                speechiness=track_data["speechiness"],
                tempo=track_data["tempo"],
                time_signature=track_data["time_signature"],
                valence=track_data["valence"],
            )
        )

    return track_list
