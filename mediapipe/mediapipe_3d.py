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
#image_path = '/home/stefan/Documents/Master/MV_Project/V2V-Pytorch/V2V-PoseNet-PyTorch/pose_estimation/img3/'
image_path = '/home/stefan/Documents/Master/Machine_Vision/images/img3/'

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


class HandPoseDetectorMediapipe():

    def __init__(self, number_hands=1, min_depth=300, max_depth=1500, depth_mode=False, live_mode=True):
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
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM, 
            result_callback=self.process_result, 
            num_hands = number_hands)

        # create a hand detector 
        self.detector = HandLandmarker.create_from_options(options)

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
                self.rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 

                # Perform the detection 
                self.start_time = datetime.now() 
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img)
                self.detector.detect_async(mp_image, timestamp.get_milliseconds())
    

    def hpe_mediapipe_saved_image(self):
        """ 
            performs hand pose estimation (with mediapipe) using saved images
        """
        if not self.busy_flag and self.next_img:
            self.busy_flag = True
            self.next_img = False
            if self.i >= 200:
                    exit()

            depth_img = np.load(image_path + 'depth/' + str(self.i) + '.npy')
            self.depth_image = depth_img
                
            # retrieve left image
            self.rgb_img = np.load(image_path + 'rgb/left/' + str(self.i) + '.npy')

            # Perform the hand detection 
            self.start_time = datetime.now()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img)
            self.detector.detect_async(mp_image, self.i)


    def process_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """ 
            processes the hand pose estimation mediapipe results (draws detected hand and saves image)
        """
        if result.hand_landmarks: 
            # get detected hand in image and world coords.
            hand_landmarks = result.hand_landmarks[0]
            world_landmarks = result.hand_world_landmarks[0]

            model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks])
            height, width = self.rgb_img.shape[:2]
            image_points = np.float32([[l.x * width, l.y * height] for l in hand_landmarks])
            
            # get calibration matrix of camera
            # calibration_params = self.get_zed_calibration_parameters()        
            # fx = calibration_params.left_cam.fx
            # fy = calibration_params.left_cam.fy
            # cx = calibration_params.left_cam.cx
            # cy = calibration_params.left_cam.cy
            
            fx = 683.8394165 
            fy = 683.8394165 
            cx = 643.83508301
            cy = 366.22180176

            camera_matrix = np.array([[fx,  0.,  cx],
                                    [0.,  fy,  cy],
                                    [0.,  0.,  1.]])
            distortion = np.zeros((4, 1))


            success, rvecs, tvecs, = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_SQPNP
            )

            # needs to 4x4 because you have to use homogeneous coordinates
            transformation = np.eye(4)
            transformation[0:3, 3] = tvecs.squeeze()
            # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

            # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate(
                (model_points, np.ones((21, 1))), axis=1)

            # apply the transformation
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
            # Transform back to non homogeneous coordinates
            world_points = world_points[:, :3] / world_points[:, 3:4]
            
            # *1000 because we have millimeters (Matthias added this minus to be in the optical_frame tf frame of ROS)
            world_points = -world_points * 1000

            # dont know why, but i need to invert the y values
            world_points[:, 1] = -world_points[:, 1]

            delta_time = datetime.now() - self.start_time

            VIZ = True
            if VIZ:
                # vis depth image
                orig_points = self.depthmap2points(self.depth_image, fx, fy)
                orig_points = orig_points.reshape(-1, 3)
                self.data_pointcloud.points = o3d.utility.Vector3dVector(orig_points)
                
                # vis detected hand
                output_pointcloud_CoM = o3d.geometry.PointCloud()
                output_pointcloud_CoM.points = o3d.utility.Vector3dVector(world_points.reshape(-1, 3))
                
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

        else:
            print("No hand detected.")

        self.busy_flag = False


    def get_zed_calibration_parameters(self):
        # Initialize the ZED camera
        if not self.zed.is_opened():
            self.open_camera()

        # Get calibration parameters
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        return calibration_params
    
    
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
    HPE = HandPoseDetectorMediapipe(number_hands=1)
    
    CAMERA = False
    if CAMERA:
        HPE.open_camera()

        while True:
            HPE.hpe_mediapipe_rgb_image()
            #HPE.display_points()
            stop = HPE.display_image()
            if stop:
                break
            time.sleep(0.005)

        HPE.close_camera()
    else:
        #open = HPE.open_camera()
        while True:
            HPE.hpe_mediapipe_saved_image()
            HPE.display_points()
            time.sleep(0.005)
