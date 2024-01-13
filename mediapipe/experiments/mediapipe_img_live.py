import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# camera imports
import pyzed.sl as sl
import cv2 
import numpy as np

# Params
CAMERA = True



# # Index of each keypoint in the hand_landmarks list
LANDMARK_INDEX_DICT = {
    "WRIST": 0,
    "THUMB_CMC": 1,  # Thumb carpometacarpal joint
    "THUMB_MCP": 2,  # Thumb metacarpophalangeal joint
    "THUMB_IP": 3,   # Thumb interphalangeal joint
    "THUMB_TIP": 4,  # Thumb tip
    "INDEX_FINGER_MCP": 5,   # Index finger metacarpophalangeal joint
    "INDEX_FINGER_PIP": 6,   # Index finger proximal interphalangeal joint
    "INDEX_FINGER_DIP": 7,   # Index finger distal interphalangeal joint
    "INDEX_FINGER_TIP": 8,   # Index finger tip
    "MIDDLE_FINGER_MCP": 9,  # Middle finger metacarpophalangeal joint
    "MIDDLE_FINGER_PIP": 10, # Middle finger proximal interphalangeal joint
    "MIDDLE_FINGER_DIP": 11, # Middle finger distal interphalangeal joint
    "MIDDLE_FINGER_TIP": 12, # Middle finger tip
    "RING_FINGER_MCP": 13,   # Ring finger metacarpophalangeal joint
    "RING_FINGER_PIP": 14,   # Ring finger proximal interphalangeal joint
    "RING_FINGER_DIP": 15,   # Ring finger distal interphalangeal joint
    "RING_FINGER_TIP": 16,   # Ring finger tip
    "PINKY_MCP": 17,         # Pinky metacarpophalangeal joint
    "PINKY_PIP": 18,         # Pinky proximal interphalangeal joint
    "PINKY_DIP": 19,         # Pinky distal interphalangeal joint
    "PINKY_TIP": 20         # Pinky tip
}

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
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
    # 0) Open camera
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    
    # Open the camera
    err = zed.open(init_params)
    #err = zed.open()
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))
    
    image_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    # 1) model path
    model_path = '/home/stefan/Documents/Master/MV_Project/mediapipe/model/hand_landmarker.task'

    # 2) create mediapipe task
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Create a hand landmarker instance with the image mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)
        
    with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.

      while True:
          if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
              # A new image is available if grab() returns ERROR_CODE.SUCCESS
              zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
              timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
              print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image_zed.get_width(), image_zed.get_height(), timestamp.get_milliseconds()))
              # Displaying the image 
              # Use get_data() to get the numpy array
              image_rgba = image_zed.get_data()
              image_rgb = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2RGB) 
              image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

              # Load the input image from a numpy array.
              mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)

          # 4) Perform the detection
          # Perform hand landmarks detection on the provided single image.
          # The hand landmarker must be created with the image mode.
          hand_landmarker_result = landmarker.detect(mp_image)
          annotated_image = draw_landmarks_on_image(image_rgb, hand_landmarker_result)
          
          cv2.imshow('image', annotated_image) 
          if cv2.waitKey(5) & 0xFF == ord('q'):
            break

