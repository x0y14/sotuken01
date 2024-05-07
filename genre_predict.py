import csv

import pandas as pd

from lib.db import *
from dataclasses import dataclass
from typing import List
from sklearn.decomposition import PCA
from io import StringIO
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


@dataclass
class AlbumWithGenre:
    id: str
    name: str
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
    genre: List[str]


@dataclass
class AlbumIsX(AlbumWithGenre):
    is_x: int


# Define
DB_NAME = "soundtrack.sqlite"

# DB Setup
conn = connect_db(DB_NAME)
create_albums_table_if_not_exists(conn)
create_tracks_table_if_not_exists(conn)
create_genres_table_if_not_exists(conn)

simplefilter("ignore", category=ConvergenceWarning)


# 1次元にした特徴データをもつジャンル付きアルバムリスト
def get_album_with_genre(connection: sqlite3.Connection) -> List[AlbumWithGenre]:
    album_with_genres: List[AlbumWithGenre] = []
    # ジャンルデータが存在するアルバムをすべて取得
    album_genres = get_genre_list(connection, 200, 0)
    for genre in album_genres:
        album_id = genre.id  # SPOTIFY_ALBUM_ID
        album = get_album(conn, album_id)
        album_name = album.name
        album_genre_tags = genre.tags
        album_tracks = get_tracks(conn, album_id, 100, 0)

        # 曲が極端に少ない場合飛ばす
        if len(album_tracks) < 3:
            continue

        # アルバム内の各楽曲のパラメータをまとめて各々スカラーに.(12xN -> 12x1)
        album_acousticness = []
        album_danceability = []
        album_duration_ms = []
        album_energy = []
        album_instrumentalness = []
        album_key = []
        album_liveness = []
        album_loudness = []
        album_mode = []
        album_speechiness = []
        album_tempo = []
        album_time_signature = []
        album_valence = []

        for track in album_tracks:
            album_acousticness.append(track.acousticness)
            album_danceability.append(track.danceability)
            album_duration_ms.append(track.duration_ms)
            album_energy.append(track.energy)
            album_instrumentalness.append(track.instrumentalness)
            album_key.append(track.key)
            album_liveness.append(track.liveness)
            album_loudness.append(track.loudness)
            album_mode.append(track.mode)
            album_speechiness.append(track.speechiness)
            album_tempo.append(track.tempo)
            album_time_signature.append(track.time_signature)
            album_valence.append(track.valence)

        album_matrix = [
            album_acousticness,
            album_danceability,
            album_duration_ms,
            album_energy,
            album_instrumentalness,
            album_key,
            album_liveness,
            album_loudness,
            album_mode,
            album_speechiness,
            album_tempo,
            album_time_signature,
            album_valence
        ]

        # (楽曲数)次元から1次元に
        pca = PCA(n_components=1)
        album_pca = pca.fit_transform(album_matrix)

        album_with_genres.append(
            AlbumWithGenre(
                id=album_id,
                name=album_name,
                acousticness=(album_pca[0])[0],
                danceability=(album_pca[1])[0],
                duration_ms=(album_pca[2])[0],
                energy=(album_pca[3])[0],
                instrumentalness=(album_pca[4])[0],
                key=(album_pca[5])[0],
                liveness=(album_pca[6])[0],
                loudness=(album_pca[7])[0],
                mode=(album_pca[8])[0],
                speechiness=(album_pca[9])[0],
                tempo=(album_pca[10])[0],
                time_signature=(album_pca[11])[0],
                valence=(album_pca[12])[0],
                genre=album_genre_tags
            )
        )
    return album_with_genres


def create_album_with_is_genre(genre: str, album: AlbumWithGenre) -> AlbumIsX:
    return AlbumIsX(
        id=album.id,
        name=album.name,
        acousticness=album.acousticness,
        danceability=album.danceability,
        duration_ms=album.duration_ms,
        energy=album.energy,
        instrumentalness=album.instrumentalness,
        key=album.key,
        liveness=album.liveness,
        loudness=album.loudness,
        mode=album.mode,
        speechiness=album.speechiness,
        tempo=album.tempo,
        time_signature=album.time_signature,
        valence=album.valence,
        genre=album.genre,
        is_x=1 if genre in album.genre else 0,
    )


def check_only_one(_list: List[int]) -> int:
    if 1 in _list and 0 in _list:
        return -1
    if 1 in _list:
        return 1
    return 0


def main():
    # 推定したいジャンル
    genre_want_to_predict = "fantasy"
    # 学習に使う行
    learn_row = 114

    # アルバムデータの用意
    album_with_genres = get_album_with_genre(conn)
    album_is_x: List[dict] = []
    genre_how_many_in_learn = 0
    genre_how_many_in_test = 0
    no = 0
    for album_with_genre in album_with_genres:
        v = vars(create_album_with_is_genre(genre_want_to_predict, album_with_genre))

        if genre_want_to_predict in album_with_genre.genre:
            if no < learn_row:
                genre_how_many_in_learn += 1
            else:
                genre_how_many_in_test += 1

        del v["genre"]
        album_is_x.append(
            v
        )
        no += 1

    buf = StringIO()

    # 教師データ
    df_csv = (pd.json_normalize(album_is_x)
              .to_csv(buf, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, header=False))
    buf.pos = 0
    df_learn = pd.read_csv(
        StringIO(buf.getvalue()),
        names=[
            "id",
            "name",
            "acousticness",
            "danceability",
            "duration_ms",
            "energy",
            "instrumentalness",
            "key",
            "liveness",
            "loudness",
            "mode",
            "speechiness",
            "tempo",
            "time_signature",
            "valence",
            "is_x",
        ],
        nrows=learn_row,
    )
    # STR列を消す
    df_learn = df_learn.drop(labels="id", axis="columns")
    df_learn = df_learn.drop(labels="name", axis="columns")
    df_learn_answers = df_learn["is_x"].tolist()
    df_learn = df_learn.drop(labels="is_x", axis="columns")

    # テストデータ
    buf.pos = 0
    df_test = pd.read_csv(
        StringIO(buf.getvalue()),
        names=[
            "id",
            "name",
            "acousticness",
            "danceability",
            "duration_ms",
            "energy",
            "instrumentalness",
            "key",
            "liveness",
            "loudness",
            "mode",
            "speechiness",
            "tempo",
            "time_signature",
            "valence",
            "is_x",
        ],
        skiprows=learn_row,
    )
    df_test_answers = df_test["is_x"].tolist()
    # 同じ分データを消す
    df_test = df_test.drop(labels="id", axis="columns")
    df_test_name = df_test["name"].tolist()
    df_test = df_test.drop(labels="name", axis="columns")
    df_test = df_test.drop(labels="is_x", axis="columns")

    print(f"作品は{genre_want_to_predict}ジャンルか?")
    print(f"教師のなかに{genre_want_to_predict}は{genre_how_many_in_learn}件、テストのなかに{genre_how_many_in_test}")

    # 学習
    # [LinearSVC]
    from sklearn.svm import LinearSVC
    model = LinearSVC(dual=True, max_iter=20000)
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("LinearSVC:", accuracy_score(df_test_answers, test_predict))
    # if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
    #     print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")
    # for predict_no in range(len(test_predict)):
    #     print(f"<{df_test_name[predict_no]}> {album_with_genres[learn_row + predict_no].genre}")
    #     print(f"  {'正解' if test_predict[predict_no] == df_test_answers[predict_no] else '不正解'}"
    #           f", 予想:{'はい' if test_predict[predict_no] == 1 else 'いいえ'}"
    #           f", 答え:{'はい' if df_test_answers[predict_no] == 1 else 'いいえ'}")

    print("---")

    # [RidgeClassifier]
    from sklearn.linear_model import RidgeClassifier
    model = RidgeClassifier()
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("RidgeClassifier:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")
    # for predict_no in range(len(test_predict)):
    #     print(f"<{df_test_name[predict_no]}> {album_with_genres[learn_row + predict_no].genre}")
    #     print(f"  {'正解' if test_predict[predict_no] == df_test_answers[predict_no] else '不正解'}"
    #           f", 予想:{'はい' if test_predict[predict_no] == 1 else 'いいえ'}"
    #           f", 答え:{'はい' if df_test_answers[predict_no] == 1 else 'いいえ'}")

    print("---")

    # [LogisticRegression]
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("LogisticRegression:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")
    # for predict_no in range(len(test_predict)):
    #     print(f"<{df_test_name[predict_no]}> {album_with_genres[learn_row + predict_no].genre}")
    #     print(f"  {'正解' if test_predict[predict_no] == df_test_answers[predict_no] else '不正解'}"
    #           f", 予想:{'はい' if test_predict[predict_no] == 1 else 'いいえ'}"
    #           f", 答え:{'はい' if df_test_answers[predict_no] == 1 else 'いいえ'}")

    print("---")

    # [SGDClassifier]
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier()
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("SGDClassifier:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")

    print("---")

    # [Perceptron]
    from sklearn.linear_model import Perceptron
    model = Perceptron(random_state=3)
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("Perceptron:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")

    print("---")

    # [PassiveAggressiveClassifier]
    from sklearn.linear_model import PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier()
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("PassiveAggressiveClassifier:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")

    print("---")

    # [SVC]
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = make_pipeline(StandardScaler(), SVC())
    model.fit(df_learn, df_learn_answers)
    test_predict = model.predict(df_test)
    print("SVC:", accuracy_score(df_test_answers, test_predict))
    if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
        print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")

    print("---")

    # # [NuSVC]
    # from sklearn.svm import NuSVC
    # model = make_pipeline(StandardScaler(), NuSVC())
    # model.fit(df_learn, df_learn_answers)
    # test_predict = model.predict(df_test)
    # print("NuSVC:", accuracy_score(df_test_answers, test_predict))
    # if check_only_one(df_test_answers) == -1 and check_only_one(test_predict) != -1:
    #     print(f"[!] まともに推測できていない可能性あり: 回答がすべて{check_only_one(test_predict)}")


if __name__ == "__main__":
    main()
