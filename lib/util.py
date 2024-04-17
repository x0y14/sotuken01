from typing import List, Any

from lib.db import Track


def create_track_vec(track: Track) -> List[float]:
    return [
        track.acousticness,
        track.danceability,
        # float(track.duration_ms),
        track.energy,
        track.instrumentalness,
        float(track.key),
        track.liveness,
        track.loudness,
        float(track.mode),
        track.speechiness,
        track.tempo,
        float(track.time_signature),
        track.valence,
    ]


def inflate_data(dd: List[List[Any]], how_many_add: int, dim: int) -> List[List[Any]]:
    result = []
    result.extend(dd)
    d = [0 for _ in range(dim)]
    for _ in range(how_many_add):
        result.append(d)
    return result
