# Start by importing all of the main libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# camera imports
import pyzed.sl as sl
import cv2 
import numpy as np
import open3d as o3d

# timing
import time
from datetime import datetime

# Image path of the stored live hand images
image_path = '/home/stefan/Documents/Master/MV_Project/V2V-Pytorch/V2V-PoseNet-PyTorch/pose_estimation/img3/'

lines = [
    [0, 1],
    [0, 5],
    [0, 9],
    [0, 13],
    [0, 17],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20]
]


class HandPoseDetectorMediapipe3D():
    """
        class for 3D Handpose detection using mediapipe and a stereo camera 
        By using Triangulation (camera intrinsics and positions) and the detected hands in the left and 
        right camera images (with mediapipe 2D) the 3D position of the hand can be determined
    """

    def __init__(self, number_hands=1, min_depth=300, max_depth=1500, depth_mode=True, live_mode=True):
        # ----- general -----
        # min and max depth of the depth image
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # depth image is only used for checking
        self.depth_image = None
        self.i = 0
        self.rgb_img_left = None
        self.rgb_img_right = None
        self.detect_right_hand = False
        
        # results of other image
        self.landmarker_left = None
        self.timestamp_left = None
        self.landmarker_right = None
        self.timestamp_right = None
        
        # for livestream mode
        self.busy_flag = False
        self.image = None
        self.img_type = "rgb"
        # to display the "run time"
        self.start_time = None

        # ----- camera -----
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
        
        # ------ mediapipe -------
        model_path = '/home/stefan/Documents/Master/MV_Project/mediapipe/model/hand_landmarker.task'

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options_left = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, 
            result_callback=self.process_result_left, 
            num_hands = number_hands)
        options_right = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, 
            result_callback=self.process_result_right, 
            num_hands = number_hands)
        
        # create a hand detector for the left and for the right image
        # (2 individual detectors needed due to the palm detection stage 
        # (is not called every time, position hand determined based on the last position))  
        self.detector_left = HandLandmarker.create_from_options(options_left)
        self.detector_right = HandLandmarker.create_from_options(options_right)

        # ------ visualization ---------
        self.points = None
        self.output_CoM = None
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.visualizer.create_window()                
        self.visualizer.register_key_callback(81, self.exit_callback)
        self.visualizer.register_key_callback(65, self.key_callback)
        self.next_img = True
        self.update = False
        self.data_pointcloud = o3d.geometry.PointCloud()
        self.line_set_CoM = o3d.geometry.LineSet()


    def hpe_mediapipe_triangulation_live_images(self):
        """
            Performs HPE with images from live camera feed
        """
        # Fill with code (like in HandPoseDetectorMediapipe only with left and right image)
        return


    def hpe_mediapipe_triangulation_stored_images(self):
        """
            Performs HPE with stored images in image_path
        """
        if not self.busy_flag and self.next_img:
            self.busy_flag = True
            self.next_img = False
            if self.i >= 200:
                    exit()

            depth_img = np.load(image_path + 'depth/' + str(self.i) + '.npy')
            self.depth_image = depth_img
                
            # retrieve left and right image
            self.rgb_img_left = np.load(image_path + 'rgb/left/' + str(self.i) + '.npy')
            self.rgb_img_right = np.load(image_path + 'rgb/right/' + str(self.i) + '.npy')

            # Perform the hand detection 
            self.start_time = datetime.now()
           
            # left hand
            mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img_left)
            self.detector_left.detect_async(mp_image_left, self.i)
            
            # rigth hand
            mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img_right)
            self.detector_right.detect_async(mp_image_right, self.i) 

        return
    

    def process_result_left(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            callback function when left handpose detector detects hands
        """
        if result.hand_landmarks: 
            self.landmarker_left = result
            self.timestamp_left = timestamp_ms
            if self.landmarker_right is not None:
                if self.timestamp_left == self.timestamp_right:
                    self.process_result()
                else:
                    print("Timestamps of detected hand in left and right image do not match. Results will be discarded.")
                # delete the prev detected landmarks
                self.landmarker_left = None
                self.timestamp_left = None
                self.landmarker_right = None
                self.timestamp_right = None
        else: 
            print("No Hand in the left image detected.")

        return


    def process_result_right(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            callback function when rigth handpose detector detects hands
        """
        if result.hand_landmarks: 
            self.landmarker_right = result
            self.timestamp_right = timestamp_ms
            if self.landmarker_left is not None:
                if self.timestamp_left == self.timestamp_right:
                    self.process_result()
                else:
                    print("Timestamps of detected hand in left and right image do not match. Results will be discarded.")
                # delete the prev detected landmarks
                self.landmarker_left = None
                self.timestamp_left = None
                self.landmarker_right = None
                self.timestamp_right = None
        else: 
            print("No Hand in the right image detected.")
        
        return
    

    def process_result(self):
        """
            function that extracts the 3d handpose out of the hand detected in the left and right image
        """
        # Check if both detected hands are left or right, if not discard the results
        if not ((self.landmarker_left.handedness[0][0].category_name) == (self.landmarker_right.handedness[0][0].category_name)):
            print("Hands handiness mismatch on both images! Results will be discarded.")
            return
        
        VIZ = False
        if VIZ:
            annotated_image_l = self.draw_landmarks_on_image(self.rgb_img_left, self.landmarker_left)
            annotated_image_r = self.draw_landmarks_on_image(self.rgb_img_right, self.landmarker_right)
            img = np.vstack((annotated_image_l, annotated_image_r))
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            cv2.imshow('Left and Right image', img) 
            cv2.waitKey(0)

        landmarks_left = self.get_x_y_img_coords_from_landmarks(self.rgb_img_left, self.landmarker_left)
        landmarks_right = self.get_x_y_img_coords_from_landmarks(self.rgb_img_right, self.landmarker_right)

        # we use fixed calibration_params, but you can extract them during runtime with the following code
        # calibration_params = self.get_zed_calibration_parameters()
        # fx = calibration_params.left_cam.fx
        # fy = calibration_params.left_cam.fy

        fx = 683.8394165039062
        fy = 683.8394165039062
        baseline = 119.916

        # determine the depth of the hand landmarks based on the landmarks_left and landmarks_right
        depth = self.stereo_depth(landmarks_left, landmarks_right, baseline, fx)

        # calculate 3d points of the hand (the left image keypoints are used because the depth image is aligned with the left image)
        img_height, img_width = self.depth_image.shape[:2]
        hand3d = self.pixel2world(landmarks_left[:, 0], landmarks_left[:, 1], depth, img_width, img_height, fx, fy)
        hand3d = np.array(hand3d).T

        delta_time = datetime.now() - self.start_time

        VIZ = True
        if VIZ:
            # vis depth image
            orig_points = self.depthmap2points(self.depth_image, fx, fy)
            orig_points = orig_points.reshape(-1, 3)
            self.data_pointcloud.points = o3d.utility.Vector3dVector(orig_points)
            
            # vis detected hand
            output_pointcloud_CoM = o3d.geometry.PointCloud()
            output_pointcloud_CoM.points = o3d.utility.Vector3dVector(hand3d.reshape(-1, 3))
            
            self.line_set_CoM.lines = o3d.utility.Vector2iVector(lines)
            self.line_set_CoM.points = output_pointcloud_CoM.points
            self.line_set_CoM.paint_uniform_color([0,1,0])
            options = self.visualizer.get_render_option()
            options.point_size = 3.0

            if self.update:
                self.visualizer.update_geometry(self.data_pointcloud)
                #self.visualizer.update_geometry(self.ref_pointcloud)
                self.visualizer.update_geometry(self.line_set_CoM)

            else:    
                self.visualizer.add_geometry(self.data_pointcloud)
                #self.visualizer.add_geometry(self.ref_pointcloud)
                self.visualizer.add_geometry(self.line_set_CoM)
                self.update = True


            print("Run Time: ", delta_time.total_seconds() * 1000, "ms")
            self.busy_flag = False
        return
    
    
    def get_zed_calibration_parameters(self):
        # Initialize the ZED camera
        if not self.zed.is_opened():
            self.open_camera()

        # Get calibration parameters
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        return calibration_params


    def stereo_depth(self, left_points, right_points, baseline, focal_length_x):
        """
            calculates the depth using the left and right image of a stereo camera
        """
        disparity = left_points[:, 0] - right_points[:, 0]
        depth = baseline * focal_length_x / disparity
        return depth


    def get_x_y_img_coords_from_landmarks(self, image, landmarks, hand_number=0):
        """
            returns the x and y image coordinates (int) of the detected landmarks of hand number 
        """
        xList = []
        yList = []
        # if at least one hand is detected return bbox, else return empty bbox
        if landmarks.hand_landmarks:
            hand_landmarks = landmarks.hand_landmarks[hand_number]
            height, width = image.shape[:2]
            #xList = np.array([[round(landmark.x * width) for landmark in hand_landmarks]], dtype=np.int32) 
            #yList = np.array([[round(landmark.y * height) for landmark in hand_landmarks]], dtype=np.int32)
            xList = np.array([[landmark.x * width for landmark in hand_landmarks]], dtype=np.float32) 
            yList = np.array([[landmark.y * height for landmark in hand_landmarks]], dtype=np.float32)
            return np.concatenate((xList, yList), axis=0).T
        else:
            return []


    def depthmap2points(self, image, fx, fy):
        """
            converts depth image into a 3D pointcloud
        """
        h, w = image.shape
        x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
        points = np.zeros((h, w, 3), dtype=np.float32)
        points[:,:,0], points[:,:,1], points[:,:,2] = self.pixel2world(x, y, image, w, h, fx, fy)
        return points


    def pixel2world(self, x, y, z, img_width, img_height, fx, fy):
        w_x = (x - img_width / 2) * z / fx
        w_y = (img_height / 2 - y) * z / fy
        w_z = z
        return w_x, w_y, w_z    
        
    
    def display_points(self):
        """
            displays/visualizes the pointsclouds
        """
        self.visualizer.poll_events()
        self.visualizer.update_renderer()


    def key_callback(self, vis):
        self.i = self.i + 10
        #self.visualizer.remove_geometry()
        self.next_img = True


    def exit_callback(self, vis):
        exit()



if __name__ == "__main__":
    TEST = False
    if TEST:
        HPE = HandPoseDetectorMediapipe3D(number_hands=1, live_mode=False)
        while True:
                HPE.test_mode()
                HPE.display_points()
                time.sleep(0.005)
    else:
        HPE = HandPoseDetectorMediapipe3D(number_hands=1)
        # determines if we use Camera or saved images as a image source
        CAMERA = False
        
        if CAMERA:
            HPE.open_camera()

            while True:
                HPE.hpe_mediapipe_triangulation_live_images()
                #HPE.display_points()
                stop = HPE.display_image()
                if stop:
                    break
                time.sleep(0.005)

            HPE.close_camera()
        else:
            #open = HPE.open_camera()
            while True:
                HPE.hpe_mediapipe_triangulation_stored_images()
                HPE.display_points()
                time.sleep(0.005)