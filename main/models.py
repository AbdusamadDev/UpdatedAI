import sqlite3
from datetime import datetime


class BaseModel(object):
    """docstring for BaseModel"""

    connection = sqlite3.connect("db.db")
    cursor = connection.cursor()

    def __init__(self, user, cam, emotion=None):
        self.user = user
        self.cam = cam
        self.emotion = emotion
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS existed_face (
                    "id" INTEGER,
                    "camera_url" TEXT,
                    "user_id" TEXT,
                    "date_created" TEXT,
                    PRIMARY KEY ('id')
                )
            """
        )
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS temp_face (
                    "id" INTEGER,
                    "camera_url" TEXT,
                    "user_id" TEXT,
                    "emotion" TEXT,
                    "date_created" TEXT,
                    PRIMARY KEY ('id')
                )
            """
        )
        self.cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS emotion_data (
                    "id" INTEGER,
                    "camera_url" TEXT,
                    "user_id" TEXT,
                    "emotion" TEXT,
                    "date_created" TEXT,
                    PRIMARY KEY ('id')
                )
            """
        )
        self.connection.commit()

    def save(self, table_name):
        now = str(datetime.now())
        if table_name == "existed_face":
            self.cursor.execute(
                f"""
                    INSERT INTO '{table_name}' (camera_url, user_id, date_created) VALUES (?, ?, ?)
                """,
                (self.cam, self.user, now),
            )
        else:
            self.cursor.execute(
                f"""
                    INSERT INTO '{table_name}' (camera_url, user_id, emotion, date_created) VALUES (?, ?, ?, ?)
                """,
                (self.cam, self.user, self.emotion, now),
            )
        self.connection.commit()
