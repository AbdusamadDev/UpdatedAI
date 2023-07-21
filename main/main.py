import logging
import time
import sqlite3
from collections import Counter, defaultdict
from face_emotion_detector import FaceEmotionDetector
from models import BaseModel

logging.basicConfig(level=logging.INFO)

DB_PATH = "db.db"
ROOT_DIR = "./media"
MODEL_WEIGHTS = "./model.h5"
NO_DATA_LIMIT = 10


class User:
    def __init__(self, id, cam_id, emotion, timestamp):
        self.id = id
        self.cam_id = cam_id
        self.start_time = timestamp
        self.last_seen = timestamp
        self.emotions = [emotion]
        self.five_sec_emotion_printed = False

    def update(self, emotion, timestamp):
        self.last_seen = timestamp
        self.emotions.append(emotion)


def get_camera_urls_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM urls")
        urls = cursor.fetchall()
    return [url[0] for url in urls]


def calculate_weighted_emotion(emotion_list):
    weights = [0.5**i for i in range(len(emotion_list))][::-1]
    weighted_emotions = Counter(
        {emotion: weights[i] for i, emotion in enumerate(emotion_list)}
    )
    total_weight = sum(weights)
    emotion_percents = {
        emotion: round(count / total_weight * 100)
        for emotion, count in weighted_emotions.items()
    }
    return emotion_percents


def main():
    camera_urls = ["http://192.168.1.52:5000/video_feed"]
    # get_camera_urls_from_db()
    detector = FaceEmotionDetector(ROOT_DIR, MODEL_WEIGHTS)
    detector.add_camera(camera_urls)
    user_dict = defaultdict(dict)
    no_data_counter = 0

    while True:
        for idx, video_capture in enumerate(detector.video_captures):
            try:
                output = next(detector.detect_and_display_faces(video_capture))
            except StopIteration:
                logging.info("Выход из цикла while")
                video_capture.release()
                break
            except Exception as e:
                logging.error(f"Ошибка при обработке видеозахвата: {e}")
                continue

            current_time = time.time()
            cam_id = camera_urls[idx]
            user_id = output["user_id"]
            emotion = output.get("emotion")

            if emotion is None or user_id == "No face detected":
                no_data_counter += 1
                if no_data_counter > NO_DATA_LIMIT:
                    break
                continue

            no_data_counter = 0
            if user_id not in user_dict[cam_id]:
                base = BaseModel(cam=cam_id, user=output["user_id"])
                base.save("existed_face")
                user_dict[cam_id][user_id] = User(
                    user_id, cam_id, emotion, current_time
                )
            else:
                user_dict[cam_id][user_id].update(emotion, current_time)

        for cam_id, users in list(user_dict.items()):
            for user_id, user in list(users.items()):
                if (
                    current_time - user.start_time > 2
                    and not user.five_sec_emotion_printed
                ):
                    user.five_sec_emotion_printed = True
                    base = BaseModel(
                        cam=user.cam_id,
                        user=user.id,
                        emotion=str(calculate_weighted_emotion(user.emotions)),
                    )
                    base.save("temp_face")

                if current_time > user.last_seen + 5 and user.five_sec_emotion_printed:
                    user = user_dict[cam_id].pop(user_id)
                    base = BaseModel(
                        cam=user.cam_id,
                        user=user.id,
                        emotion=str(calculate_weighted_emotion(user.emotions)),
                    )
                    base.save("emotion_data")


if __name__ == "__main__":
    main()
