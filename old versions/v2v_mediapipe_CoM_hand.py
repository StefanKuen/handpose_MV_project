# Start by importing all of the main libraries
import numpy as np
import torch
#torch.multiprocessing.set_start_method('spawn')# good solution !!!!
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
import pathlib
import open3d as o3d
import time
import torch.multiprocessing as multiprocessing
import struct

# Add the root directory so we can use the following imports
root_directory	=	str(pathlib.Path(__file__).parent.parent.resolve()).replace("\\","/")
sys.path.append(root_directory)

from src.v2v_model import V2VModel
from src.v2v_util_pointcloud import V2VVoxelization

# Add the mediapipe directory, so that we can add the mediapipe HandPoseDetectorMediapipe class
# you could also simply copy the mediapipe_live_stream.py module into the current directory
mediapipe_dir = '/home/stefan/Documents/Master/MV_Project/mediapipe'
sys.path.append(mediapipe_dir)
from mediapipe_live_stream import HandPoseDetectorMediapipe
import mediapipe as mp
import pyzed.sl as sl

# new imports
from scipy import stats, ndimage
import cv2
import imutils 
from scipy.optimize import minimize
# timing
import time
from datetime import datetime


image_path = '/home/stefan/Documents/Master/MV_Project/V2V-Pytorch/V2V-PoseNet-PyTorch/pose_estimation/img3/'


# def points2pixels(points, img_width, img_height, fx, fy):
#     pixels = np.zeros((points.shape[0], 2))
#     pixels[:, 0], pixels[:, 1] = \
#         world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
#     return pixels


# def world2pixel(x, y, z, img_width, img_height, fx, fy):
#     p_x = x * fx / z + img_width / 2
#     p_y = img_height / 2 - y * fy / z
#     return p_x, p_y


# def pointcloudpoints_to_depthimg_coords(points, fx, fy, img_width, img_height):
#     """
#         Transforms 3D pointcloud points into image coordinates (y, x) with a depth value(z)
#         for example: for detected 3D hand pose landmarks to image landmarks
#     """
    
#     # Assume points is a numpy array with shape (N, 3) representing N 3D points (x, y, z)
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]

#     # Project points onto 2D plane (x, y)
#     x_2d = (fx * x / z) + img_width / 2
#     y_2d = img_height / 2 - (fy * y / z)

#     # Create depth image
#     depth_image = np.zeros((img_width, img_height))

#     # Clip points outside the resolution
#     valid_points = (x_2d >= 0) & (x_2d < img_width) & (y_2d >= 0) & (y_2d < img_height)

#     # Assign depth values to the depth image
#     depth_image[y_2d[valid_points].astype(int), x_2d[valid_points].astype(int)] = z[valid_points]

#     return np.array((x_2d[valid_points].T, y_2d[valid_points].T, z[valid_points].T), dtype=np.uint32)


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


class HandPoseDetectorV2V(HandPoseDetectorMediapipe):
    """
        class for Handpose detection with V2V and mediapipe (as hand detector)
    """
    def __init__(self, number_hands=1, min_depth=300, max_depth=1500):
        super().__init__(number_hands, min_depth, max_depth, depth_mode=True)
        self.min_depth = min_depth
        self.max_depth = max_depth
        # current depth image
        self.depth_image = None
        # needed to replay saved images
        self.i = 0

        # needed for back transformation
        self.rotation = 0
        self.bbox = np.array([0, 0, 0, 0])

        # load v2v model
        model_path = root_directory + "/output/cvpr15_MSRAHandGestureDB/model.pt"
        self.model = torch.load(model_path);
        self.model.eval()
        
        # model to gpu (cuda) if avaible
        self.device = None
        if torch.cuda.is_available():
            self.device  =   torch.device('cuda')  
        else:
            self.device  =   torch.device('cpu')
        
        self.dtype = torch.float
        self.model = self.model.to(self.device, self.dtype)

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
        self.ref_pointcloud = o3d.geometry.PointCloud()
        self.line_set_CoM = o3d.geometry.LineSet()

    def hpe_v2v_rgbdepth_live_image(self):
        """ 
            performs hand pose estimation (with v2v and mediapipe) using an 
            rgb image(mediapipe) and a depth image (v2v) from the camera
        """
        if self.busy_flag == False:
            self.busy_flag = True
            image_zed = sl.Mat()
            depth_zed = sl.Mat()
            runtime_parameters = sl.RuntimeParameters()

            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
                self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH) # Retrieve depth matrix. Depth is aligned on the left RGB image
                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
                
                # Use get_data() to get the numpy array
                image_bgra = image_zed.get_data()
                depth_image = depth_zed.get_data()

                # Transform color image
                image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR) 
                # Transform BGR image (OpenCV) to RGB image, because mediapipe expects RGB as input image
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 

                # Adjust depth image
                # Filter out nan, -inf, inf values
                depth_image = np.where(np.isnan(depth_image), self.max_depth, depth_image)
                depth_image = np.where(np.isneginf(depth_image), self.min_depth, depth_image)
                depth_image = np.where(np.isinf(depth_image), self.max_depth, depth_image)
                # save current depth image
                self.depth_image = depth_image

                # Perform the hand detection 
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                self.start_time = datetime.now()
                self.landmarker.detect_async(mp_image, timestamp.get_milliseconds())


    def hpe_v2v_saved_images(self):
        """
            performs hand pose estimation (with v2v and mediapipe) using an 
            rgb image(mediapipe) and a depth image (v2v) using saved depth and bgr images
        """
        if not self.busy_flag and self.next_img:
            if self.i >= 250:
                exit()
            self.busy_flag = True
            self.next_img = False
            depth_img = np.load(image_path + 'depth/' + str(self.i) + '.npy')
            rgb_img = np.load(image_path + 'rgb/left/' + str(self.i) + '.npy')
            #self.i = self.i + 1
            print("Image number: ", self.i)
            self.depth_image = depth_img
            # Perform the hand detection 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            self.start_time = datetime.now()
            self.landmarker.detect_async(mp_image, self.i)


    def process_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            overwrite the process_result function, perform V2V with the detected hands from mediapipe
        """
        image_bgr = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR) 
        self.img_type = "depth"

        # if at least one hand is detected  call bbox and rotation
        if result.hand_landmarks: 
            # --------- use the mediapipe landmarks too extract hand ------------- 
            # get x and y image-coords of the detected landmarks
            landmarks_xy = self.get_x_y_landmarks(image_bgr, result)

            # get the rotation of the hand
            rotation = self.get_rotation_hand_palm(landmarks_xy)

            # get bbox of the hand
            bbox, center = self.get_bbox_hand(self.depth_image, landmarks_xy)
            
            # visualize bbox
            #cv2.rectangle(self.depth_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            #image_hand = self.depth_image

            # prepare the hand for v2v network (cut hand out with bbox and also in depth)
            image_hand = self.cut_hand_out(landmarks_xy, bbox)
            
            VIS = False
            if VIS:
                img_bgr = cv2.cvtColor(image_hand/np.max(image_hand), cv2.COLOR_GRAY2BGR) 
                img_bgr[center[1], center[0], :] = [0, 0, 255]
                cv2.imshow('color', img_bgr) 
                cv2.waitKey(0)

            # calculate the center of mass
            minDepth = np.min(image_hand)
            maxDepth = np.max(image_hand)
            center_of_mass = self.calculateCoM(image_hand, minDepth, maxDepth)
            # print("center_of_mass: ", center_of_mass)

            # -------- Transform to world coordinates --------
            # # Get camera intrinsics
            #calibration_params = self.zed.get_camera_information().camera_configuration.calibration_parameters
            # # depth image is aligned on the left RGB image, therefore retrieve intrinsics of the left camera
            #fx = calibration_params.left_cam.fx
            #fy = calibration_params.left_cam.fy
            
            # fixed values for recorded images, can be retrieved with the above code
            fx = 683.6222534179688
            fy = 683.6222534179688

            # calculate pointcloud of hand
            hand_world = self.depthmap2points(image_hand, fx, fy)
            hand_world = hand_world.reshape(-1, 3)
            
            # transform Center of mass to world coordinates
            img_height, img_width = self.depth_image.shape
            CoM_world = self.pixel2world(center_of_mass[0], center_of_mass[1], center_of_mass[2], img_width, img_height, fx, fy)
            CoM_world = np.array(CoM_world)
            
            # transform center of bbox to world coords.
            center_bbox_world = self.pixel2world(center[0], center[1], self.depth_image[center[1], center[0]], img_width, img_height, fx, fy)
            center_bbox_world = np.array(center_bbox_world)

            # transform bbox to world coords
            leftup_bbox_world = self.pixel2world(bbox[0], bbox[1], self.depth_image[center[1], center[0]], img_width, img_height, fx, fy)
            leftup_bbox_world = np.array(leftup_bbox_world)
            rightdown_bbox_world = self.pixel2world(bbox[2], bbox[3], self.depth_image[center[1], center[0]], img_width, img_height, fx, fy)
            rightdown_bbox_world = np.array(rightdown_bbox_world)

            # cut hand with transf. bbox out
            mask_hand1 = np.where(hand_world[:, 0] > leftup_bbox_world[0], True, False)
            mask_hand2 = np.where(hand_world[:, 1] < leftup_bbox_world[1], True, False)
            mask_hand3 = np.where(hand_world[:, 0] < rightdown_bbox_world[0], True, False)
            mask_hand4 = np.where(hand_world[:, 1] > rightdown_bbox_world[1], True, False)
            mask = mask_hand1 * mask_hand2 * mask_hand3 * mask_hand4
            hand_world = hand_world[mask, :] 

            # --------- Detect the hand -----------
            # rotate the hand so that it is vertical
            hand_world = self.rotate_pointcloud(hand_world, center_bbox_world, -rotation)

            # Voxelize data
            cubic_size = 200
            v2v_voxelization = V2VVoxelization(cubic_size=cubic_size, original_size=100, augmentation=True)
            voxelized_points_CoM = v2v_voxelization.voxelize(hand_world, CoM_world)

            # Use model to estiamte v2v pose
            torch_input_CoM = torch.from_numpy(voxelized_points_CoM)
            nn_input_CoM  = torch_input_CoM.to(self.device, self.dtype)
            nn_output_CoM  = self.model(nn_input_CoM[None, ...])
            nn_output_CoM  = nn_output_CoM.cpu().detach().numpy()
            self.output_CoM = v2v_voxelization.evaluate(nn_output_CoM, [CoM_world])[0]
            
            # rotate detected hand back
            self.output_CoM =  self.rotate_pointcloud(self.output_CoM, center_bbox_world, rotation)
            print("hand landmarks:", self.output_CoM)

            delta_time = datetime.now() - self.start_time

            orig_points = self.depthmap2points(self.depth_image, fx, fy)
            orig_points = orig_points.reshape(-1, 3)

            # --------- Visualize --------------
            # viz bbox hand
            #img = rot_img.copy()
            #cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            #self.image = img
            
            # If True: Visualize the detected hand in the original image
            # If False: Visualize the detected hand in the cutout image
            VIS_ORIGINAL_IMAGE = True

            if VIS_ORIGINAL_IMAGE:
                self.data_pointcloud.points = o3d.utility.Vector3dVector(orig_points)

                # vis detected hand
                hand_pointcloud = o3d.geometry.PointCloud()
                hand_pointcloud.points = o3d.utility.Vector3dVector(self.output_CoM)
                
                self.line_set_CoM.lines = o3d.utility.Vector2iVector(lines)
                self.line_set_CoM.points = hand_pointcloud.points
                self.line_set_CoM.paint_uniform_color([0,1,0])
                options = self.visualizer.get_render_option()
                options.point_size = 3.0

                if self.update:
                    self.visualizer.update_geometry(self.data_pointcloud)
                    self.visualizer.update_geometry(self.ref_pointcloud)
                    self.visualizer.update_geometry(self.line_set_CoM)

                else:    
                    self.visualizer.add_geometry(self.data_pointcloud)
                    self.visualizer.add_geometry(self.ref_pointcloud)
                    self.visualizer.add_geometry(self.line_set_CoM)
                    self.update = True
            else:
                # visualize only the hand
                hand_world = self.rotate_pointcloud(hand_world, center_bbox_world, rotation)
                self.points = hand_world
                self.data_pointcloud.points = o3d.utility.Vector3dVector(self.points)

                # vis bbox
                #self.ref_pointcloud.points = o3d.utility.Vector3dVector(np.concatenate((center_bbox_world, leftup_bbox_world, rightdown_bbox_world)).reshape(3, 3))
                #self.ref_pointcloud.colors = o3d.utility.Vector3dVector(np.array([[1,0,0], [1,0,0], [1,0,0]]).reshape(3, 3))
                # vis center of mass
                self.ref_pointcloud.points = o3d.utility.Vector3dVector(CoM_world.reshape(1, 3))
                self.ref_pointcloud.colors = o3d.utility.Vector3dVector(np.array([1,0,0]).reshape(1, 3))
                
                # vis detected hand
                output_pointcloud_CoM = o3d.geometry.PointCloud()
                output_pointcloud_CoM.points = o3d.utility.Vector3dVector(self.output_CoM)
                
                self.line_set_CoM.lines = o3d.utility.Vector2iVector(lines)
                self.line_set_CoM.points = output_pointcloud_CoM.points
                self.line_set_CoM.paint_uniform_color([0,1,0])
                options = self.visualizer.get_render_option()
                options.point_size = 3.0

                if self.update:
                    self.visualizer.update_geometry(self.data_pointcloud)
                    self.visualizer.update_geometry(self.ref_pointcloud)
                    self.visualizer.update_geometry(self.line_set_CoM)

                else:    
                    self.visualizer.add_geometry(self.data_pointcloud)
                    self.visualizer.add_geometry(self.ref_pointcloud)
                    self.visualizer.add_geometry(self.line_set_CoM)
                    self.update = True

            print("Run Time: ", delta_time.total_seconds() * 1000, "ms")


        self.busy_flag = False


    def rotate_pointcloud(self, points, rotation_center, rotation_angle_degrees):
        """
            rotates pointcloud points around z-axis for a fixed rotation angle around a given rotation center
        """
        # Convert rotation angle to radians
        rotation_angle_radians = np.radians(rotation_angle_degrees)

        # Translate points to the origin
        translated_points = points - rotation_center

        # Define rotation matrix
        rotation_matrix = np.array([[np.cos(rotation_angle_radians), -np.sin(rotation_angle_radians), 0],
                                    [np.sin(rotation_angle_radians), np.cos(rotation_angle_radians), 0],
                                    [0, 0, 1]])

        # Apply rotation matrix to points
        rotated_points = np.dot(rotation_matrix, translated_points.T).T

        # Translate points back to the original position
        rotated_points += rotation_center

        return rotated_points


    def get_x_y_landmarks(self, image, landmarks, hand_number=0):
        """
            returns the x and y image coordinates (int) of the detected landmarks of hand number 
        """
        xList = []
        yList = []
        # if at least one hand is detected return bbox, else return empty bbox
        if landmarks.hand_landmarks:
            hand_landmarks = landmarks.hand_landmarks[hand_number]
            height, width = image.shape[:2]
            xList = np.array([[round(landmark.x * width) for landmark in hand_landmarks]], dtype=np.int32) 
            yList = np.array([[round(landmark.y * height) for landmark in hand_landmarks]], dtype=np.int32)
            return np.concatenate((xList, yList), axis=0).T
        else:
            return []

    
    def get_bbox_hand(self, image, landmarks_xy, deltax = 20, deltay = 20, hand_number=0):
        """
            Extracts the bbox (and center) of the detected hand inside the image
        """
        height, width = image.shape[:2]
        xList = landmarks_xy[:, 0]
        yList = landmarks_xy[:, 1] 
        # fix when xmin, ymin or xmax, ymax gets out of image size +- deltax, deltay
        xList = np.where(xList < deltax, deltax, xList)
        yList = np.where(yList < deltax, deltay, yList)
        xList = np.where(xList > width-1-deltax, width-1-deltax, xList)
        yList = np.where(yList > height-1-deltay, height-1-deltay, yList)
        bbox = np.array([np.min(xList)-deltax, np.min(yList)-deltay, np.max(xList)+deltax, np.max(yList)+deltay], dtype=np.uint32)
        center = np.array([bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2], dtype=np.uint32)
        return bbox, center



    def get_rotation_hand_middlefinger(self, image, landmarks, hand_number=0):
        """
             Extracts the rotation of the detected hands inside the image
             Takes the middle finger as reference point, alignes the middlefinger
             mcp with vertical [0, -1]
        """
        # if at least one hand is detected return bbox, else return empty bbox
        if landmarks.hand_landmarks:
            # get the reference points (wrist, middle_mcp)
            hand_landmarks = landmarks.hand_landmarks[hand_number]
            h, w, _ = image.shape
            wrist = np.array([hand_landmarks[0].x * w, hand_landmarks[0].y * h])
            middle_mcp = np.array([hand_landmarks[9].x * w, hand_landmarks[9].y * h])
            
            # calculate the vectors (reference(vertical) and wrist to middle)
            vec = middle_mcp - wrist
            vec = vec/np.linalg.norm(vec)
            vertical = np.array([0, -1])
            
            # calculate the angle
            angle = np.arccos(np.clip(np.dot(vec, vertical), -1.0, 1.0))
            if vec[0] > 0:
                angle = 2*np.pi-angle
                #angle = -angle
            return angle/np.pi*180
        else:
            return 0
        

    def get_rotation_hand_palm(self, landmarks_xy, hand_number=0):
        """
             Extracts the rotation of the detected hands inside the image
             Takes the palm as reference point, minimizes the distance between
             pinky to vertical [0, -1] and index to vertical by rotating the hand.
        """
        # get the reference points (wrist, index_mcp, pinky_mcp)
        wrist = landmarks_xy[0, :]
        index_mcp = landmarks_xy[5, :]
        pinky_mcp = landmarks_xy[17, :]
        
        # calculate the vectors (reference(fixed) and palm)
        v_f = np.array([0, -1]).reshape(2, 1)
        v_1 = (index_mcp - wrist).reshape(2, 1)
        # works better without the norm
        #v_1 = v_1/np.linalg.norm(v_1)
        v_2 = (pinky_mcp - wrist).reshape(2, 1)
        #v_2 = v_2/np.linalg.norm(v_2)
    

        # Initial guess for rotation angle (in radians)
        initial_rotation_angle = 0.0
        # Optimize the rotation angle
        result = minimize(self._optimizer_function, initial_rotation_angle, args=(v_f, v_1, v_2), method='BFGS')
        # Extract the optimized rotation angle
        optimized_rotation_angle = result.x[0]
        return optimized_rotation_angle/np.pi*180   
    


    def _optimizer_function(self, rotation_angle, v_f, v_1, v_2):
        """
            function for the get_rotation_hand_palm(...) optimization
        """
        # Construct the rotation matrix
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]]).reshape(2, 2)

        rotated_v1 = np.dot(rotation_matrix, v_1)
        rotated_v2 = np.dot(rotation_matrix, v_2)

        distance_1 = np.linalg.norm(v_f - rotated_v1)
        distance_2 = np.linalg.norm(v_f - rotated_v2)

        return distance_1 + distance_2


    def cut_hand_out(self, rot_landmarks, bbox, delta_depth_hand = 100):
        """
            function that prepares the depth image for the v2v network
        """
        # cut hand out with bbox
        img_hand = np.ones_like(self.depth_image)*self.max_depth
        img_hand[bbox[1]:bbox[3], bbox[0]:bbox[2]] = self.depth_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # extract the hand (use depth of the middle of the middle finger and make +- depth bbox around that)
        # get depth of the middle of the middle finger (should approx. be the depth center of the hand)
        depth_hand = self.depth_image[rot_landmarks[10, 1], rot_landmarks[10, 0]]
        # set everything that is outside this depth bbox to max_depth (this will not be taken into account by V2V)
        img_hand = np.where(img_hand < (depth_hand - delta_depth_hand), self.max_depth, img_hand)
        img_hand = np.where(img_hand > (depth_hand + delta_depth_hand), self.max_depth, img_hand)

        return img_hand


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
    

    def calculateCoM(self, dpt, MinDepth, MaxDepth):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass (image coord.)
        """

        dc = dpt.copy()
        dc[dc <= MinDepth] = 0
        dc[dc >= MaxDepth] = 0
        cc = ndimage.center_of_mass(dc > 0)
        num = np.count_nonzero(dc)
        com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float32)

        if num == 0:
            return np.array((0, 0, 0), np.float32)
        else:
            return com/num
        

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
    HPE = HandPoseDetectorV2V(number_hands=2)
    
    # determines if we use Camera or saved images as a image source
    CAMERA = False
    
    if CAMERA:
        HPE.open_camera()

        while True:
            HPE.hpe_v2v_rgbdepth_live_image()
            #HPE.display_points()
            stop = HPE.display_image()
            if stop:
                break
            time.sleep(0.005)

        HPE.close_camera()
    else:
        while True:
            HPE.hpe_v2v_saved_images()
            HPE.display_points()
            time.sleep(0.005)