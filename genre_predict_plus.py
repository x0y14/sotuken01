import csv

import pandas as pd
import scipy

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


@dataclass
class AlbumWithoutGenre:
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


LEARN_ROW = 114


def create_model(album_with_genres, genre: str):
    models = []

    album_is_x: List[dict] = []
    for album_with_genre in album_with_genres:
        # if "animation" in album_with_genre.genre:
        #     album_with_genre.genre.remove("animation")
        v = vars(create_album_with_is_genre(genre, album_with_genre))
        del v["genre"]
        album_is_x.append(v)

    buf = StringIO()

    # 教師データ
    pd.json_normalize(album_is_x).to_csv(buf, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL, header=False)
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
        nrows=LEARN_ROW,
    )
    # 答えの保存
    df_learn_answers = df_learn["is_x"].tolist()
    # STR列を消す
    df_learn = df_learn.drop(labels="id", axis="columns")
    df_learn = df_learn.drop(labels="name", axis="columns")
    df_learn = df_learn.drop(labels="is_x", axis="columns")
    # 標準化
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(df_learn)
    # scaler.transform(df_learn)
    # df_learn = pd.DataFrame(scaler.transform(df_learn), columns=df_learn.columns)
    # df_learn = df_learn.apply(scipy.stats.zscore, axis=0)

    # 学習
    # LinearSVC
    # from sklearn.svm import LinearSVC
    # model = LinearSVC(dual=True, max_iter=20000)
    # model.fit(df_learn, df_learn_answers)
    # models.append(model)

    # RidgeClassifier
    from sklearn.linear_model import RidgeClassifier
    model = RidgeClassifier()
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # LogisticRegression
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # SGDClassifier
    # from sklearn.linear_model import SGDClassifier
    # model = SGDClassifier()
    # model.fit(df_learn, df_learn_answers)
    # models.append(model)

    # Perceptron
    from sklearn.linear_model import Perceptron
    model = Perceptron(random_state=0, shuffle=True, eta0=0.1)
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # PassiveAggressiveClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    model = PassiveAggressiveClassifier()
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = make_pipeline(StandardScaler(), SVC())
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # KNeighborsClassifier
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    # RadiusNeighborsClassifier
    # from sklearn.neighbors import RadiusNeighborsClassifier  # RadiusNeighborsClassifierのインポート
    # model = RadiusNeighborsClassifier()
    # model.fit(df_learn, df_learn_answers)
    # models.append(model)

    # NearestCentroid
    from sklearn.neighbors import NearestCentroid
    model = NearestCentroid()
    model.fit(df_learn, df_learn_answers)
    models.append(model)

    return models


def remove_album_genre(album: AlbumWithGenre) -> AlbumWithoutGenre:
    return AlbumWithoutGenre(
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
    )


def jadge(percent_list: List[float]):
    return majority_vote(percent_list)
    # return percent_list[1] > 0.6 or percent_list[2] > 0.6
    # return percent_list[2] > 0.6 and percent_list[0] > 0.6


def majority_vote(percent_list: List[float]) -> bool:
    threshold = 0.7
    half = int(len(percent_list) / 2)
    up_vote = 0
    for percent in percent_list:
        if percent > threshold:
            up_vote += 1
    return up_vote >= half


def jaccard(data1, data2):
    items = 0
    for item in data1:
        if item in data2:
            items += 1
    return items / (len(data1) + len(data2) - items)


def main():
    album_with_genres = get_album_with_genre(conn)
    # モデルの準備
    comedy_model = create_model(album_with_genres, "comedy")
    horror_model = create_model(album_with_genres, "horror")
    crime_model = create_model(album_with_genres, "crime")
    biography_model = create_model(album_with_genres, "biography")
    romance_model = create_model(album_with_genres, "romance")
    family_model = create_model(album_with_genres, "family")
    adventure_model = create_model(album_with_genres, "adventure")
    history_model = create_model(album_with_genres, "history")
    thriller_model = create_model(album_with_genres, "thriller")
    action_model = create_model(album_with_genres, "action")
    music_model = create_model(album_with_genres, "music")
    drama_model = create_model(album_with_genres, "drama")
    animation_model = create_model(album_with_genres, "animation")
    sport_model = create_model(album_with_genres, "sport")
    documentary_model = create_model(album_with_genres, "documentary")
    mystery_model = create_model(album_with_genres, "mystery")
    scifi_model = create_model(album_with_genres, "sci-fi")
    fantasy_model = create_model(album_with_genres, "fantasy")
    # short_model = create_model(album_with_genres, "short")
    # western_model = create_model(album_with_genres, "western")
    models = {
        "comedy": comedy_model,
        "horror": horror_model,
        "crime": crime_model,
        "biography": biography_model,
        "romance": romance_model,
        "family": family_model,
        "adventure": adventure_model,
        "history": history_model,
        "thriller": thriller_model,
        "action": action_model,
        "music": music_model,
        "drama": drama_model,
        "animation": animation_model,
        "sport": sport_model,
        "documentary": documentary_model,
        "mystery": mystery_model,
        "sci-fi": scifi_model,
        "fantasy": fantasy_model,
        # "short": short_model,  #
        # "western": western_model,  #
    }
    model_names = [
        "LinearSVC",
        "RidgeClassifier",
        "LogisticRegression",
        "SGDClassifier",
        "Perceptron",
        "PassiveAggressiveClassifier",
        "SVC"
    ]

    test_datas = album_with_genres[LEARN_ROW:]
    total_jaccard_score = 0.0
    for test_data in test_datas:
        if len(get_tracks(conn, test_data.id, 100, 0)) < 3:
            continue
        print(f"<{test_data.name}>")
        print(f"  actual: {','.join(test_data.genre)}")
        # create test data
        buf = StringIO()
        pd.json_normalize(vars(remove_album_genre(test_data))).to_csv(buf, index=False, encoding='utf-8',
                                                                      quoting=csv.QUOTE_ALL,
                                                                      header=False)
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
            ],
        )
        df_test = df_test.drop(labels="id", axis="columns")
        df_test = df_test.drop(labels="name", axis="columns")
        # 標準化
        # df_test = df_test.apply(scipy.stats.zscore, axis=0)
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # scaler.fit(df_test)
        # scaler.transform(df_test)
        # df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

        predicted_genres = []

        for genre in models:
            percents = []
            for model in models[genre]:
                predict_result = model.predict(df_test)
                # print(f"{model}: {predict_result}")
                percents.append(predict_result)
            if jadge(percents):
                predicted_genres.append(genre)
        print(f"  predicted: {','.join(predicted_genres)}")
        score = jaccard(test_data.genre, predicted_genres)
        print(f"  score:{score}")
        total_jaccard_score += score

    print(f"avg: {total_jaccard_score / len(test_datas)}")


if __name__ == "__main__":
    main()
