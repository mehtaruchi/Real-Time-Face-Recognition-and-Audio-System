import math
import cv2
from sklearn import neighbors,svm
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pyttsx3
import time
from app import Log,db
import subprocess
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
# import the necessary packages
from threading import Thread

import datetime
'''

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()



class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
    	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

'''




engine=pyttsx3.init()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
names = ["Anuj","Harsh","Jainam","Kunjal","Mugdha","Prachiti","Ruchi","Vidhi"]

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            print("Training image" + str(img_path))
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes,num_jitters=20)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    #knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance',metric='euclidean')
    svm_clf = svm.SVC(kernel="linear",probability=True,verbose=True)
    #knn_clf.fit(X, y)
    svm_clf.fit(X,y)
    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            #pickle.dump(knn_clf, f)
            pickle.dump(svm_clf, f)

    #return knn_clf
    return svm_clf

def predict(X_img_path, svm_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an Unknown Person person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'Unknown Person' will be returned.
    

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    """
    #Load a trained KNN model (if one was passed in)
    if svm_clf is None:
        with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)
    
    # Load image file and find face locations
    X_img =face_recognition.load_image_file(X_img_path)
    #X_img = X_img_path
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample = 2)
    print(len(X_face_locations))
    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    probs_faces = svm_clf.predict_proba(faces_encodings)
    predictions = []
    print(probs_faces)
    probs_faces = probs_faces.tolist()
    for prob_face in probs_faces:          
        
            
        val, idx = max((val, idx) for (idx, val) in enumerate(prob_face))         
    
        if val < 0.4:
            predictions.append("Unknown Person")
        else:
            predictions.append(names[idx])
    # Use the KNN model to find the best matches for the test face
    #closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=3)
    return predictions
    # are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(predictions)
    # # Predict classes and remove classifications that aren't within the threshold
    # return [(pred, loc) if rec else ("Unknown Person", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    print(predictions)
    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))



    # Display the resulting image
    pil_image.show()


def preprocess(frame):


    return face_locations,face_encodings


  


if __name__ == "__main__":




    process_this_frame = True
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training SVM classifier...")
    #classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=3) #MODEL IS HERE
    print("Training complete!")
    for image_file in os.listdir("dataset/friends"):
        full_file_path = os.path.join("dataset/friends", image_file)

        print("Looking for faces in {}".format(image_file))

        #Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        print(predictions)
        # # Print results on the console
        # for name, (top, right, bottom, left) in predictions,X_face_locations:
        #     print("- Found {} at ({}, {})".format(name, left, top))

        # # # Display results overlaid on an image
        # show_prediction_labels_on_image(os.path.join("dataset/friends", image_file), predictions)

    # video_capture = cv2.VideoCapture(0)
    # video_capture.set(3,1920)
    # video_capture.set(4,1080)

    '''
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    cv2.waitKey(50)
    ctr=0
    c = 0

    
    # STEP 2: Using the trained classifier, make predictions for Unknown Person images
    while True:
    # Grab a single frame of video
    
        # ret, frame = video_capture.read()
        #cv2.imshow('Video', frame)
        #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = vs.read()
	    frame = imutils.resize(frame, width=400)
        rgb_frame = frame[:, :, ::-1]



    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    #if process_this_frame:
        # Find all the faces and face encodings in the current frame of video

        #process_this_frame = not process_this_frame

 
        
        predictions = predict(frame, model_path="trained_knn_model.clf")
            

        mul_name_list=[]
        flag = 0
        # for name, (top, right, bottom, left) in predictions:
        #     if(name=="Unknown Person"):
        #         ctr=ctr+1
                
        #     """if(ctr<=1 and name=="Unknown Person"):
        #         #print('Unknown Person recalculating')
        #         flag = 1
        #         continue
        #     """
        #     ctr=0
        #     complete_name=name.split(' ')
        #     #user=Log(first_name=complete_name[0],last_name=complete_name[1])
        #     #db.session.add(user)
        #     #db.session.commit()
        #     mul_name_list.append(name)
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        #     print("- Found {} at ({}, {})".format(name, left, top))

        for name in predictions:
            mul_name_list.append(name)
        lts=' and '.join(map(str,mul_name_list))
        print('welcome to the blockchain lab'+lts)    
            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
                       

        #if(flag==1):
        #    continue
        
            
        cv2.imshow('Video', frame)
        fps.update()
        if(len(lts)!=0):
            ps = subprocess.Popen(['python', 'speak.py', 'Welcome to the Blockchain Lab '+ lts], stdout=subprocess.PIPE)

        time.sleep(3)

    # Display the resulting image      
        

    # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
fps.stop()
cv2.destroyAllWindows()
vs.stop()

    '''
