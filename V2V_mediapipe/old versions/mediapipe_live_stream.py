import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# camera imports
import pyzed.sl as sl
import cv2 
import numpy as np

# timing
import time
from datetime import datetime


class HandPoseDetectorMediapipe():

    def __init__(self, number_hands=1, min_depth=300, max_depth=1500, depth_mode=False, live_mode=True):
        # for livestream mode
        self.busy_flag = False
        self.image = None
        self.img_type = "rgb"
        # to display the "run time"
        self.start_time = None

        # camera
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.sdk_verbose = 1
        if depth_mode:
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        else:
            self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA 
        self.init_params.depth_minimum_distance = min_depth
        self.init_params.depth_maximum_distance = max_depth
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
        
        # mediapipe
        model_path = '/home/stefan/Documents/Master/MV_Project/mediapipe/model/hand_landmarker.task'
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if live_mode:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM, 
                result_callback=self.process_result, 
                num_hands = number_hands)
        else:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE, 
                num_hands = number_hands)
            
        self.landmarker = HandLandmarker.create_from_options(options)


    def open_camera(self):
        """ 
        opens the Zed camera
        """
        # open the camera
        err = self.zed.open(self.init_params)
        
        # return false if open was not successful
        if err != sl.ERROR_CODE.SUCCESS:
            return False
        
        # print serial number (for checking)
        zed_serial = self.zed.get_camera_information().serial_number
        print("Hello! This is my serial number: {0}".format(zed_serial))

        return True
    

    def close_camera(self):
        """ 
        closes the Zed camera
        """
        self.zed.close()


    def hpe_mediapipe_rgb_image(self):
        """ 
            performs hand pose estimation (with mediapipe) using an rgb image (from camera)
        """
        if self.busy_flag == False:
            self.busy_flag = True
            image_zed = sl.Mat()
            runtime_parameters = sl.RuntimeParameters()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
                
                # Use get_data() to get the numpy array
                image_bgra = image_zed.get_data()
                image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR) 
                # Transform BGR image (OpenCV) to RGB image, because mediapipe expects RGB as input image
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 

                # Perform the detection 
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                # image.flags.writeable = False
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #print(timestamp.get_milliseconds())
                self.start_time = datetime.now()
                self.landmarker.detect_async(mp_image, timestamp.get_milliseconds())


    def process_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """ 
            processes the hand pose estimation mediapipe results (draws detected hand and saves image)
        """
        # transform back to BGR for OpenCV
        delta_time = datetime.now() - self.start_time
        image_bgr = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR) 
        annotated_image = self.draw_landmarks_on_image(image_bgr, result)
        self.image = annotated_image
        print("Run Time: ", delta_time.total_seconds() * 1000, "ms")
        self.busy_flag = False


    def display_image(self):
        """
            displays the image with the detected hand
        """
        if self.image is not None:
            
            if self.img_type == "depth":
                self.image = self.image/np.max(self.image)*255
            
            img = np.array(self.image, dtype=np.uint8)
            cv2.imshow('image', img) 
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return True
        
        return False


    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """
            draws the detected hand onto the current image
        """
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
        
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
    

if __name__ == "__main__":
    HPE = HandPoseDetectorMediapipe(number_hands=2)
    HPE.open_camera()

    while True:
        HPE.hpe_mediapipe_rgb_image()
        stop = HPE.display_image()
        if stop:
            break
        time.sleep(0.005)

    HPE.close_camera()
