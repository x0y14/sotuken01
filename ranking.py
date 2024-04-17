import numpy as np

from lib.db import *
from lib.util import *

DB_NAME = "soundtrack.sqlite"
WANT_TO_KNOW_MOST_SIMILAR_WITH = "2ufkFJsK2Hh2ZdmgrRmCv3"
DIM = 12


def main():
    # DBの準備
    conn = connect_db(DB_NAME)
    create_albums_table_if_not_exists(conn)
    create_tracks_table_if_not_exists(conn)

    # album_identifiers = []  # ID, NAME
    album_vecs = []

    # 実際にベクトルを作る!
    albums = get_album_list(conn, limit=30, offset=0)
    for album in albums:
        # album_identifiers.append((album.id, album.name))
        track_vecs = []
        tracks = get_tracks(conn, album_id=album.id, limit=50, offset=0)
        for track in tracks:
            track_vecs.append(create_track_vec(track))
        album_vecs.append((album.id, album.name, track_vecs))

    # ベクトルのサイズを揃える!
    # 最大探す
    maximum = 0
    for album_vec in album_vecs:
        if len(album_vec[2]) > maximum:
            maximum = len(album_vec[2])
    album_vec_same_sizes = []  # これ、最終的に使います!!
    # ぜろ埋め
    for album_vec in album_vecs:
        album_vec_same_sizes.append(
            (album_vec[0],
             album_vec[1],
             inflate_data(album_vec[2], maximum - len(album_vec[2]), DIM))
        )

    # 検索対象を探す
    position_of_search_target = 0
    for album_vec_same_size in album_vec_same_sizes:
        if album_vec_same_size[0] != WANT_TO_KNOW_MOST_SIMILAR_WITH:
            position_of_search_target += 1
        else:
            break

    print(
        f"検索対象: 「{album_vec_same_sizes[position_of_search_target][1]}」({album_vec_same_sizes[position_of_search_target][0]})")

    # まず検索対象だけNPARRAY化
    search_target = np.array(
        album_vec_same_sizes[position_of_search_target][2])

    # ソート用データの準備
    distances = []

    for album_vec_same_size in album_vec_same_sizes:
        distance = np.linalg.norm(
            np.array(album_vec_same_size[2]) - search_target)
        distances.append(
            (album_vec_same_size[0],
             album_vec_same_size[1],
             distance)
        )

    sorted_distances = sorted(distances, key=lambda x: x[2])
    rank = 1
    print()
    for sorted_distance in sorted_distances:
        print(
            f"{'[同一] ' if sorted_distance[2] == 0.0 else ''}{rank}位「{sorted_distance[1]}」({sorted_distance[0]}), 距離: {sorted_distance[2]}")
        rank += 1


if __name__ == "__main__":
    main()
