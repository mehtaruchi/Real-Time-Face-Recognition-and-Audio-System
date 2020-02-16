from app import db
class User(db.Model):

    __tablename__='users'

    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    time_stamp = db.Column(db.Integer)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)

    def __init__(self,first_name,last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.time_stamp = time.time()

    def __repr__(self):
        return f"{self.id}, {self.time_stamp}, {self.first_name} {self.last_name}"