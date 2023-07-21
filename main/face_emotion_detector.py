import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from facenet_pytorch import MTCNN, InceptionResnetV1
import datetime
import threading


class FaceEmotionDetector:
    def __init__(self, root_dir, model_weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.mtcnn = MTCNN(device=self.device)
            self.resnet = (
                InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            )
        except Exception as e:
            print(f"Ошибка при инициализации моделей: {e}")
            raise

        self.emotion_labels = [
            "jahldorlik",
            "behuzur",
            "havotir",
            "hursandchilik",
            "g'amgin",
            "hayron",
            "neytral",
        ]
        self.emotion_model = self.create_emotion_model()
        try:
            self.emotion_model.load_weights(model_weights)
        except Exception as e:
            print(f"Не удалось загрузить веса модели: {e}")
            raise

        try:
            self.known_face_encodings, self.known_face_names = self.load_face_encodings(
                root_dir
            )
        except Exception as e:
            print(f"Ошибка при загрузке кодировок лиц: {e}")
            raise

        self.emotion = ""
        self.user_id = ""
        self.video_captures = []

    def add_camera(self, urls):
        for url in urls:
            try:
                video_capture = cv2.VideoCapture(url)
                if not video_capture.isOpened():
                    print(f"Не удалось открыть захват видео для {url}")
                    continue
                else:
                    print(f"Захват видео для {url} открыт успешно")
                    self.video_captures.append(video_capture)
            except Exception as e:
                print(f"Ошибка при открытии видеозахвата для {url}: {e}")
                continue

    def start_all_video_captures(self):
        threads = []
        for video_capture in self.video_captures:
            try:
                thread = threading.Thread(
                    target=self.detect_and_display_faces, args=(video_capture,)
                )
                thread.start()
                threads.append(thread)
            except Exception as e:
                print(f"Ошибка при создании потока: {e}")

        for thread in threads:
            try:
                thread.join()
            except Exception as e:
                print(f"Ошибка при ожидании завершения потока: {e}")

    def create_emotion_model(self):
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[0], "GPU")
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(f"Ошибка при настройке GPU: {e}")

        emotion_model = tf.keras.models.Sequential()
        emotion_model.add(
            tf.keras.layers.Conv2D(
                64, (5, 5), activation="relu", input_shape=(48, 48, 1)
            )
        )
        emotion_model.add(
            tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2))
        )
        emotion_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        emotion_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        emotion_model.add(
            tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))
        )
        emotion_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        emotion_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
        emotion_model.add(
            tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))
        )
        emotion_model.add(tf.keras.layers.Flatten())
        emotion_model.add(tf.keras.layers.Dense(1024, activation="relu"))
        emotion_model.add(tf.keras.layers.Dropout(0.2))
        emotion_model.add(tf.keras.layers.Dense(1024, activation="relu"))
        emotion_model.add(tf.keras.layers.Dropout(0.2))
        emotion_model.add(tf.keras.layers.Dense(7, activation="softmax"))
        return emotion_model

    def load_face_encodings(self, root_dir):
        known_face_encodings = []
        known_face_names = []
        if "media" not in os.listdir(os.getcwd()):
            os.makedirs("media")
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".jpg") or file_name.endswith(".png"):
                        image_path = os.path.join(dir_path, file_name)
                        try:
                            image = cv2.imread(image_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            boxes, _ = self.mtcnn.detect(image)  # type: ignore
                        except Exception as e:
                            print(
                                f"Не удалось обработать изображение {image_path}: {e}"
                            )
                            continue  # пропуск этого изображения и переход к следующему

                        if boxes is not None:
                            box = boxes[0].astype(int)
                            face = image[box[1] : box[3], box[0] : box[2]]
                            if face.size == 0:
                                continue
                            face = cv2.resize(face, (160, 160))
                            face = face / 255.0  # type: ignore
                            face = (
                                torch.tensor(face.transpose((2, 0, 1)))
                                .float()
                                .to(self.device)
                                .unsqueeze(0)
                            )
                            embedding = (
                                self.resnet(face).detach().cpu().numpy().flatten()
                            )

                            known_face_encodings.append(embedding)
                            known_face_names.append(dir_name)
        return known_face_encodings, known_face_names

    def detect_and_display_faces(self, video_capture):
        while True:
            try:
                ret, frame = video_capture.read()
            except Exception as e:
                print(f"Не удалось прочитать кадр: {e}")
                continue  # пропуск этого кадра и переход к следующему

            try:
                boxes, _ = self.mtcnn.detect(frame)  # type: ignore
            except Exception as e:
                pass
                boxes = None

            if boxes is not None:
                for box in boxes:
                    box = box.astype(int)
                    face = frame[box[1] : box[3], box[0] : box[2]]
                    if face.size == 0:
                        continue
                    face = cv2.resize(face, (160, 160))
                    face = face / 255.0
                    face = (
                        torch.tensor(face.transpose((2, 0, 1)))
                        .float()
                        .to(self.device)
                        .unsqueeze(0)
                    )
                    embedding = self.resnet(face).detach().cpu().numpy().flatten()

                    distances = np.linalg.norm(
                        self.known_face_encodings - embedding, axis=1
                    )
                    argmin = distances.argmin()
                    min_distance = distances[argmin]

                    if min_distance < 1:
                        name = self.known_face_names[argmin]
                        face_gray = cv2.cvtColor(
                            frame[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY
                        )
                        face_gray = cv2.resize(face_gray, (48, 48))
                        face_gray = face_gray / 255.0  # type: ignore
                        face_gray = np.reshape(face_gray, (1, 48, 48, 1))
                        emotion = self.emotion_labels[
                            np.argmax(self.emotion_model.predict(face_gray, verbose=0))
                        ]  # type: ignore
                        yield {
                            "time": str(datetime.datetime.now()).split(".")[0],
                            "user_id": name,
                            "emotion": emotion,
                        }
            else:
                yield {
                    "time": str(datetime.datetime.now()).split(".")[0],
                    "user_id": "No face detected",
                    "emotion": "None",
                }
