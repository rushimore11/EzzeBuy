import json
import os
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "data_set", "users.json")


class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


def _seed_users():
    return {
        "admin": User(id=1, username="admin", password_hash=generate_password_hash("admin123"))
    }


def load_users():
    if not os.path.exists(USERS_FILE):
        return _seed_users()

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            raw_users = json.load(file)

        loaded_users = {}
        for record in raw_users:
            loaded_users[record["username"]] = User(
                id=record["id"],
                username=record["username"],
                password_hash=record["password_hash"],
            )

        if not loaded_users:
            return _seed_users()

        return loaded_users
    except (json.JSONDecodeError, KeyError, TypeError):
        return _seed_users()


def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

    payload = [
        {
            "id": user.id,
            "username": user.username,
            "password_hash": user.password_hash,
        }
        for user in users.values()
    ]

    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


users = load_users()

if "admin" not in users:
    users["admin"] = _seed_users()["admin"]
