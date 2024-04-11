import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util
import tensorflow as tf
from tensorflow import keras
import numpy as np
# import cv2
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps 



import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')



data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)
# from test import test

model = load_model("keras_Model.h5", compile=False)


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    image=cv2.resize(image,(int(image.shape[0]*3/4),image.shape[0]))
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    
    print("Prediction cost {:.2f} s".format(test_speed))
    return label




class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        # self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        # self.logout_button_main_window.place(x=750, y=300)

        # self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
        #                                                             self.register_new_user, fg='black')
        # self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        img=frame
        image_PIL = Image.fromarray(img)
        image = ImageOps.fit(image_PIL, size, Image.Resampling.LANCZOS)
        # img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        # class_name = class_names[index]
        confidence_score = prediction[0][index]

        # print("Class:", class_name[2:], end="")
        print("index",index)
        # print("Confidence Score:", confidence_score)


        label=test(
            image=img,
            model_dir=r"C:\Users\vibgyor\Desktop\New folder\face-attendance-system\Silent-Face-Anti-Spoofing-master\resources\anti_spoof_models",
            device_id=0
        )
        print("Label: ",label)


        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        img=self.most_recent_capture_arr
        image_PIL = Image.fromarray(img)
        image = ImageOps.fit(image_PIL, size, Image.Resampling.LANCZOS)
        # img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        # class_name = class_names[index]
        confidence_score = prediction[0][index]

        # print("Class:", class_name[2:], end="")
        print("index",index)
        print("Confidence Score:", confidence_score)

        # label = test(
        #         image=self.most_recent_capture_arr,
        #         model_dir='/home/phillip/Desktop/todays_tutorial/27_face_recognition_spoofing/code/face-attendance-system/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
        #         device_id=0
        #         )

        # if label == 1:

        #     name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        #     if name in ['unknown_person', 'no_persons_found']:
        #         util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        #     else:
        #         util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
        #         with open(self.log_path, 'a') as f:
        #             f.write('{},{},in\n'.format(name, datetime.datetime.now()))
        #             f.close()

        # # else:
        #     util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    # def logout(self):

    #     # label = test(
    #     #         image=self.most_recent_capture_arr,
    #     #         model_dir='/home/phillip/Desktop/todays_tutorial/27_face_recognition_spoofing/code/face-attendance-system/Silent-Face-Anti-Spoofing/resources/anti_spoof_models',
    #     #         device_id=0
    #     #         )

    #     # if label == 1:

    #         name = util.recognize(self.most_recent_capture_arr, self.db_dir)

    #         if name in ['unknown_person', 'no_persons_found']:
    #             util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
    #         else:
    #             util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
    #             with open(self.log_path, 'a') as f:
    #                 f.write('{},{},out\n'.format(name, datetime.datetime.now()))
    #                 f.close()

    #     # else:
    #         util.msg_box('Hey, you are a spoofer!', 'You are fake !')


    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
