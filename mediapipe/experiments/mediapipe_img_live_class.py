import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# camera imports
import pyzed.sl as sl
import cv2 
import numpy as np


class HandPoseDetectorMediapipe():

    def __init__(self):
        # camera
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.sdk_verbose = 1
        #self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA 
        self.init_params.depth_minimum_distance = 30
        self.init_params.depth_maximum_distance = 5000
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
        
        # mediapipe
        model_path = '/home/stefan/Documents/Master/MV_Project/mediapipe/model/hand_landmarker.task'
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE, 
            num_hands = 2)
        
        self.landmarker = HandLandmarker.create_from_options(options)


    def open_camera(self):
        """ opens the Zed camera
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
        self.zed.close()


    def get_rgb_image(self):
        image_zed = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
            timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
            
            # Use get_data() to get the numpy array
            image_rgba = image_zed.get_data()
            image_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB) 
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Perform the detection 
            # NOTE: WHY does it work better with BGR image 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)
            return mp_image, timestamp.get_milliseconds()


    def perform_hpe(self, mp_image: mp.Image, timestamp_ms: int):
        result = self.landmarker.detect(mp_image)
        rgb_image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_BGR2RGB)
        annotated_image = self.draw_landmarks_on_image(rgb_image, result)
        return result, annotated_image


    def draw_landmarks_on_image(self, rgb_image, detection_result):
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
    HPE = HandPoseDetectorMediapipe()
    HPE.open_camera()
    
    while True:
        mp_image, timestamp = HPE.get_rgb_image()
        result, annotated_image = HPE.perform_hpe(mp_image, timestamp)
        cv2.imshow('image', annotated_image) 
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    HPE.close_camera()
