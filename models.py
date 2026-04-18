from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


# simple demo user
users = {
    "admin": User(id=1, username="admin", password="admin123")
}