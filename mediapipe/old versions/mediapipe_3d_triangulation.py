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

from mediapipe_live_stream import HandPoseDetectorMediapipe

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


class HandPoseDetectorMediapipe3D(HandPoseDetectorMediapipe):
    """
        class for 3D Handpose detection using mediapipe and a stereo camera 
        By using Triangulation (camera intrinsics and positions) and the detected hands in the left and 
        right camera images (with mediapipe 2D) the 3D position of the hand can be determined
    """

    def __init__(self, number_hands=1, min_depth=300, max_depth=1500, depth_mode=True, live_mode=True):
        super().__init__(number_hands, min_depth, max_depth, depth_mode, live_mode)
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # depth image is only used for checking
        self.depth_image = None
        self.i = 0
        self.rgb_img_left = None
        self.rgb_img_right = None
        self.detect_right_hand = False
        
        # results of other image
        self.prev_landmarker = None
        self.prev_timestamp = None

        # visualization
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


    def test_mode(self):
        
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
            landmarks_left = self.landmarker.detect(mp_image_left)
            # rigth hand
            mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img_right)
            landmarks_right = self.landmarker.detect(mp_image_right)

            VIZ = False
            if VIZ:
                annotated_image_l = self.draw_landmarks_on_image(self.rgb_img_left, landmarks_left)
                annotated_image_r = self.draw_landmarks_on_image(self.rgb_img_right, landmarks_right)
                img = np.vstack((annotated_image_l, annotated_image_r))
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
                cv2.imshow('Left and Right image', img) 
                cv2.waitKey(0)

            #calibration_params = self.get_zed_calibration_parameters()
            landmarks_left = self.get_x_y_img_coords_from_landmarks(self.rgb_img_left, landmarks_left)
            landmarks_right = self.get_x_y_img_coords_from_landmarks(self.rgb_img_right, landmarks_right)
            
            baseline = 119.916
            #fx = calibration_params.left_cam.fx
            #fy = calibration_params.left_cam.fy
            fx = 683.8394165039062
            fy = 683.8394165039062
            # determine the depth of the hand landmarks based on the landmarks_left and landmarks_right
            depth = self.stereo_depth(landmarks_left, landmarks_right, baseline, fx)

            # calculate 3d points of the hand (the left image keypoints are used because the depth image is aligned with the left image)
            img_height, img_width = self.depth_image.shape[:2]
            hand3d = self.pixel2world(landmarks_left[:, 0], landmarks_left[:, 1], depth, img_width, img_height, fx, fy)
            #hand3d = self.pixel2world(landmarks_left[:, 0], landmarks_left[:, 1], self.depth_image[landmarks_left[:, 1].astype(np.int32), landmarks_left[:, 0].astype(np.int32)], img_width, img_height, fx, fy)
            hand3d = np.array(hand3d).T

            # other approach (DOES NOT WORK PROPERLY)
            # hand3d = self.stereo_triangulation(landmarks_left, landmarks_right, calibration_params)
            # hand3d = hand3d*1000
            # print(hand3d)

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



    def hpe_mediapipe_triangulation_live_images(self):
        # Fill with code that extracts images from live camera feed (like in HandPoseDetectorMediapipe only with left and right image)
        return
    

    def hpe_mediapipe_triangulation_saved_images(self):
        """
            performs hand pose estimation (mediapipe and triangulation) using both
            rgb stereo images (the depth image is only used for checking)
        """
        if not self.busy_flag:
            if self.i >= 250:
                exit()
            self.busy_flag = True
            print("Image number: ", self.i)

            # save depth image for hand pose checking later
            depth_img = np.load(image_path + 'depth/' + str(self.i) + '.npy')
            self.depth_image = depth_img
            
            # retrieve left and right image
            self.rgb_img_left = np.load(image_path + 'rgb/left/' + str(self.i) + '.npy')
            self.rgb_img_right = np.load(image_path + 'rgb/right/' + str(self.i) + '.npy')

            # Perform the hand detection 
            mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img_left)
            

            self.start_time = datetime.now()
            self.landmarker.detect_async(mp_image_left, self.i*10)

        elif self.detect_right_hand:
            self.detect_right_hand = False
            self.rgb_img_right = np.load(image_path + 'rgb/right/' + str(self.i) + '.npy')
            mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img_right)
            self.landmarker.detect_async(mp_image_right, self.i*10+1)


    def process_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            overwrite the process_result function, performs triangulation with the detected hands from mediapipe
        """
        print(result.hand_landmarks)
        print(timestamp_ms)
        image_bgr = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR) 

        # check if both detected hands[0] are both left or both right
        if result.hand_landmarks: 

            if self.prev_landmarker is None:
                self.prev_landmarker = result
                self.detect_right_hand = True
                print("1")
                return
            # else:
            #     # prev detected landmarks was from left stereo camera image
            #     if timestamp_ms == (self.prev_timestamp + 1):
            #         landmarks_left = self.prev_landmarker
            #         landmarks_right = result
            #         self.prev_landmarker = None
            #         print("2")
            #     # prev detected landmarks was from right stereo camera image
            #     elif timestamp_ms == (self.prev_timestamp - 1):
            #         landmarks_right = self.prev_landmarker
            #         landmarks_left = result
            #         self.prev_landmarker = None
            #         print("3")
            #     # prev detected landmarks was from another image (not same time stamp), therefore prev landmarker will be discarded
            #     else:
            #         self.prev_landmarker = result
            #         self.prev_timestamp = timestamp_ms
            #         print("Something went wrong!")
            #         return
            print("2")
            landmarks_left = self.prev_landmarker
            landmarks_right = result

            VIZ = True
            if VIZ:
                annotated_image_l = self.draw_landmarks_on_image(self.rgb_img_left, landmarks_left)
                annotated_image_r = self.draw_landmarks_on_image(self.rgb_img_right, landmarks_right)
                img = np.vstack((annotated_image_l, annotated_image_r))
                img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
                cv2.imshow('Left and Right image', img) 
                cv2.waitKey(0)


            # Get camera intrinsics
            calibration_params = self.get_zed_calibration_parameters()
            landmarks_left = self.get_x_y_img_coords_from_landmarks(image_bgr, landmarks_left)
            landmarks_right = self.get_x_y_img_coords_from_landmarks(image_bgr, landmarks_right)
            
            points_3d = self.stereo_triangulation(landmarks_left, landmarks_right, calibration_params)
            #points_3d[:, 2] = np.absolute(points_3d[:, 2])
            print(points_3d)



            # -------- DOES NOT WORK --------
            if False:
                # fixed values for recorded images, can be retrieved with the above code
                # or from calibration file in /usr/local/zed/settings/
                fx = 683.6222534179688
                fy = 683.6222534179688
                # distance between the cameras in mm
                baseline_camera = 119.916
                print(self.get_x_y_img_coords_from_landmarks(image_bgr, landmarks_left))
                print(self.get_x_y_img_coords_from_landmarks(image_bgr, landmarks_right))

                # get the depth of the landmarks
                depth_landmarks = self.stereo_depth(self.get_x_y_values_from_landmarks(landmarks_left), 
                                                    self.get_x_y_values_from_landmarks(landmarks_right), 
                                                    baseline_camera, 
                                                    0.000669484)
                print(depth_landmarks)
            # align the landmarks depth with the detced landmarks in the left image (because depth image is also aligned with left image) 
            #landmarks_xyz = landmarks_xy_left

            delta_time = datetime.now() - self.start_time

            VIZ = True
            if VIZ:
                # vis depth image
                print(calibration_params.left_cam.fx)
                orig_points = self.depthmap2points(self.depth_image, calibration_params.left_cam.fx, calibration_params.left_cam.fy)
                orig_points = orig_points.reshape(-1, 3)
                self.data_pointcloud.points = o3d.utility.Vector3dVector(orig_points)
                
                # vis detected hand
                output_pointcloud_CoM = o3d.geometry.PointCloud()
                output_pointcloud_CoM.points = o3d.utility.Vector3dVector(points_3d.reshape(-1, 3))
                
                self.line_set_CoM.lines = o3d.utility.Vector2iVector(lines)
                self.line_set_CoM.points = output_pointcloud_CoM.points
                self.line_set_CoM.paint_uniform_color([0,1,0])
                options = self.visualizer.get_render_option()
                options.point_size = 3.0

                if self.update:
                    self.visualizer.update_geometry(self.data_pointcloud)
                    #self.visualizer.update_geometry(self.ref_pointcloud)
                    self.visualizer.update_geometry(self.line_set_CoM)
                    print("1")

                else:    
                    self.visualizer.add_geometry(self.data_pointcloud)
                    #self.visualizer.add_geometry(self.ref_pointcloud)
                    self.visualizer.add_geometry(self.line_set_CoM)
                    self.update = True
                    print("2")


            print("Run Time: ", delta_time.total_seconds() * 1000, "ms")
        
        print("ende")
        self.prev_landmarker = None
        #self.busy_flag = False

    
    def get_zed_calibration_parameters(self):
        # Initialize the ZED camera
        if not self.zed.is_opened():
            self.open_camera()

        # Get calibration parameters
        calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters

        return calibration_params


    def stereo_triangulation(self, left_points, right_points, calibration_params):
        # Create projection matrices for left and right cameras
        P1 = np.hstack((calibration_params.left_cam.fx, 0, calibration_params.left_cam.cx, 0,
                        0, calibration_params.left_cam.fy, calibration_params.left_cam.cy, 0,
                        0, 0, 1, 0)).reshape((3, 4))
        baseline = 119.916
        P2 = np.hstack((calibration_params.right_cam.fx, 0, calibration_params.right_cam.cx, baseline,
                        0, calibration_params.right_cam.fy, calibration_params.right_cam.cy, 0,
                        0, 0, 1, 0)).reshape((3, 4))

        # Triangulate points
        points_3d_homogeneous = cv2.triangulatePoints(P1, P2, left_points.T, right_points.T)

        # Convert homogeneous coordinates to 3D coordinates
        points_3d = (points_3d_homogeneous[:3] / points_3d_homogeneous[3]).T

        return points_3d


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
        

    def get_x_y_z_values_from_landmarks(self, landmarks, hand_number=0):
        """
            returns the x and y values (distances in mm) of the detected landmarks of hand number 
        """
        xList = []
        yList = []
        zList = []
        # if at least one hand is detected return bbox, else return empty bbox
        if landmarks.hand_landmarks:
            hand_landmarks = landmarks.hand_landmarks[hand_number]
            xList = np.array([[landmark.x for landmark in hand_landmarks]], dtype=np.float32) 
            yList = np.array([[landmark.y for landmark in hand_landmarks]], dtype=np.float32)
            zList = np.array([[landmark.z for landmark in hand_landmarks]], dtype=np.float32)
            return np.concatenate((xList, yList, zList), axis=0).T
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
    TEST = True
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
                HPE.hpe_mediapipe_triangulation_saved_images()
                HPE.display_points()
                time.sleep(0.005)