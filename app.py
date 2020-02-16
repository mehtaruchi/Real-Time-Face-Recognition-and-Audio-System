from flask import Flask,render_template,Response,jsonify,request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from camera import VideoCamera
from datetime import datetime
import time,os,shutil,glob
from training import train
from face_recognition_knn import recognize
import subprocess
import schedule,time
import scipy
import math
import dlib
import cv2
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import pyttsx3
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

val = True
video_capture = None
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

user_first_name=None
user_last_name=None 
user_timestamp=None


app.config['SECRET_KEY']='some_key'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db = SQLAlchemy(app)
Migrate(app,db)


class User(db.Model):

    __tablename__='users'

    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    time_stamp = db.Column(db.Integer)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)
    path = db.Column(db.Text)

    def __init__(self,first_name,last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.time_stamp = time.time()

    def __repr__(self):
        return f"{self.id}, {self.time_stamp}, {self.first_name} {self.last_name}"


class Log(db.Model):

    __tablename__ = 'logs'

    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    time_stamp = db.Column(db.Integer)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)
    path = db.Column(db.Text)

    def __init__(self,first_name,last_name):
        self.first_name=first_name
        self.last_name=last_name
        self.time_stamp=time.time()

    def __repr__(self):
        return f"{self.id},{self.time_stamp}, {self.first_name} {self.last_name}"


db.create_all() 


video_capture = None
global_frame = None

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.53):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = X_img_path
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample = 2)

    # If no faces are found in the image, return an empty result.

    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown person", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

set_old = set()

def flush_set():
    global set_old
    print("Flushing")
    print(set_old)
    set_old.clear()
    print("Flushed")
    print(set_old)


@app.route('/')
def home_page():
    return render_template('homepage.html')

@app.route('/enroll')
def enroll_page():
    global val
    # print("Enroll clicked")
    val = False
    return render_template('enrollpage.html')

@app.route('/identify')
def identify_page():
    return render_template('identifypage.html')


@app.route('/start_detecting',methods=['GET'])
def start_detecting():
    # print("Something")
    global video_capture
    global val
    val = True
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    face_detector = dlib.get_frontal_face_detector()
    if(video_capture==None):
        video_capture = VideoCamera()
    
    schedule.every(0.3).minutes.do(flush_set)
    global set_old
    while val==True:

        schedule.run_pending()

        ret, frame = video_capture.cap.read()
        rgb_frame =frame[:, :, ::-1] 
        

        print("Frame taken")

        detected_faces = face_detector(rgb_frame, 1)
        face_locations = face_recognition.face_locations(rgb_frame)

        predictions = predict(frame, model_path="trained_knn_model.clf")
        print(predictions)

        set_new = set()
        set_temp = set()
        for name, (top, right, bottom, left) in predictions:
            
            print("- Found {} at ({}, {})".format(name, left, top))
            print('\n \n')
            #engine.say('welcome to the blockchain lab'+name)
            #engine.runAndWait()
            set_new.add(name)

        # print("Old and new sets")
        # print(set_old)
        # print(set_new)
        temp = set_new.difference(set_old)
        # print(temp)
        new_temp = list(temp)
        print("Final list")
        print(new_temp)
        lts=' and '.join(map(str,new_temp))
        set_old = set_new
        #### SPEAKING CODE
        if(len(new_temp)!=0):
            ps = subprocess.Popen(['python', 'speak.py', 'Welcome to the Blockchain Lab '+ lts], stdout=subprocess.PIPE)
            for name in new_temp: 
                if name == "unknown":
                    name = "unknown person"               
                complete_name=name.split(' ')

                user=Log(first_name=complete_name[0],last_name=complete_name[1])
                db.session.add(user)
                db.session.commit()
            


        #print('welcome to the blockchain lab'+lts) 


        # Display the resulting image
        #cv2.imshow('Video', rgb_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    set_old.clear()
    cv2.destroyAllWindows()
    return '<h1>Hey!</h1>'


@app.route('/record_status', methods=['POST'])
def record_status():
    print('record status called')
    global video_capture
    if video_capture == None:
        video_capture = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_capture.start_record(0)
        response = jsonify(result="started")
        response.max_age=1
        return response
    else:
        video_capture.stop_record(0)
        response= jsonify(result="stopped")
        response.max_age=1
        return response



def video_stream():
    global video_capture 
    global global_frame

    if video_capture == None:
        video_capture = VideoCamera()
        
    while True:
        frame = video_capture.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/welcome')
def welcome_page():
    user_first_name = request.args.get('first')
    user_last_name = request.args.get('last')
    if(not os.path.isdir('./'+user_first_name+' '+user_last_name)):
        os.mkdir('./knn_examples/train/'+user_first_name+' '+user_last_name)
    new_images = glob.glob('./'+'*.png')
    for im in new_images:
        shutil.move(im,os.path.join('./knn_examples/train/'+user_first_name+' '+user_last_name+'/'+os.path.basename(im)))
    new_user = User(user_first_name,user_last_name)
    db.session.add(new_user)
    db.session.commit()
    new_log = Log(user_first_name,user_last_name)
    db.session.add(new_log)
    db.session.commit()
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=3) #MODEL IS HERE
    print("Training complete!")
    return render_template('welcome_page.html', first=user_first_name,last=user_last_name)


@app.route('/logpage')
def log_page():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    start_ts = datetime.timestamp(datetime.strptime(start_date,'%d-%m-%Y')) 
    end_ts = datetime.timestamp(datetime.strptime(end_date,'%d-%m-%Y'))+24*60*60

    #all_users = Log.query.all()
    # print(datetime.fromtimestamp(start_ts)
    all_users = Log.query.filter(Log.time_stamp>=start_ts,Log.time_stamp<=end_ts)
    new_list = list()
    for user in all_users:
        user.time_stamp = str(datetime.fromtimestamp(user.time_stamp))
        new_list.append(user)

    return render_template('logpage.html',list=new_list)


if (__name__=='__main__'):
    app.run(threaded=True, debug=True)
