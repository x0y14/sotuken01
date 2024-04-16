import sqlite3
from dataclasses import dataclass


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
