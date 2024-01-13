import cv2

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


import numpy as np

import rospy
from leap_motion.msg import leapros

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo

from one_euro_filter import OneEuroFilter

from scipy.spatial.transform import Rotation as R
import time

import tf2_ros
import tf2_geometry_msgs
import geometry_msgs
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA



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

class MediaPipeHands():

    def __init__(self, camera_name):
        self.pub_ros   = rospy.Publisher('leapmotion/data',leapros, queue_size=1)

        base_options = mp.tasks.BaseOptions
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        hand_landmarker_result = mp.tasks.vision.HandLandmarkerResult
        vision_running_mode = mp.tasks.vision.RunningMode


        options = hand_landmarker_options(
            base_options=base_options(model_asset_path='mediapipe_ros/src/hand_landmarker.task'),
            running_mode=vision_running_mode.LIVE_STREAM,
            result_callback=self.process_result)
        
        self.landmarker = hand_landmarker.create_from_options(options)


        self.draw_mediapipe = True

        # Camera matrix for Oak-D Pro
        # TODO: Better use ROS and get the camera matrix from there
        self.camera_matrix = np.array([[1569.889,      0.,       960.3314],
                                [0.,      1567.9027,   533.32117],
                                [0.,         0.,        1.,]])
        #self.camera_matrix = np.eye(3)
        self.distortion = np.zeros((4, 1))
        # Found those with OpenCV calibration: [[ 0.08089285  0.00510587  0.00050131 -0.00228297 -0.3632532 ]]

        self.fov_x = np.rad2deg(2 * np.arctan2(1569.889, 2 * 1569.889))


        self.cv_bridge = CvBridge()

        self.br = tf2_ros.TransformBroadcaster()
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)

        self.filter_index_tip = None
        self.filter_index_mcp = None
        self.filter_thumb_tip = None
        self.filter_thumb_mcp = None
        self.filter_wrist = None

            
        # For webcam input:
        #self.cap = cv2.VideoCapture(0)

        self.camera_info_sub = rospy.Subscriber(camera_name+"/camera_info", CameraInfo, self.camera_info_cb)
        rospy.sleep(.5)
        self.image_raw_sub = rospy.Subscriber(camera_name+"/image_raw", Image, self.image_cb)

        print(camera_name+"/camera_info")

        print("Started Node")
    
    # Create a hand landmarker instance with the live stream mode:
    def print_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        print('hand landmarker result: {}'.format(result))

    def camera_info_cb(self, msg):

        for i in range(3):
            for k in range(3):
                self.camera_matrix[i][k] = msg.K[i*3+k]
                print(i*3+k)

        # for i in range(5):
        #     self.distortion = msg.D[i]
        self.image_height = msg.height
        self.image_width = msg.width
        self.annotated_img = np.zeros(shape=(self.image_height, self.image_width))
        self.camera_info_sub.unregister()
        print("Got Camera Parameters")
        print(self.camera_matrix)

    def image_cb(self, msg):

        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            get_image_flag = True
            #print(msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9)
            self.image_timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
            
        except CvBridgeError as e:
            print(e)


        # while self.cap.isOpened() and not rospy.is_shutdown():
        #     success, image = self.cap.read()
        #     if not success:
        #         print("Ignoring empty camera frame.")
        #         # If loading a video, use 'break' instead of 'continue'.
        #         continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarker.detect_async(mp_image, mp.Timestamp.from_seconds(self.image_timestamp).microseconds())

        cv2.imshow('MediaPipe Hands', self.annotated_img)
        #cv2.imwrite("MediaPipeHands.png", annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            rospy.is_shutdown = True


    def process_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # Draw the hand annotations on the image.
        #image.flags.writeable = True
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #frame_height, frame_width, channels = image.shape
        world_points_total = []
        quat_total = []
        #print(result)



        hand_landmarks_list = result.hand_landmarks
        world_landmarks_list = result.hand_world_landmarks

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            world_landmarks = world_landmarks_list[idx]


        # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # hand_landmarks_proto.landmark.extend([
        # landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        # ])
        # if self.draw_mediapipe:
        #     self.mp_drawing.draw_landmarks(
        #         image,
        #         hand_landmarks,
        #         self.mp_hands.HAND_CONNECTIONS,
        #         self.mp_drawing_styles.get_default_hand_landmarks_style(),
        #         self.mp_drawing_styles.get_default_hand_connections_style())

            # print(output_image.numpy_view().shape)
            # annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
            # cv2.imwrite('MediaPipeHands.png', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # if cv2.waitKey(100) & 0xFF == 27:
            #     rospy.is_shutdown = True

            model_points = np.float32([[-l.x, -l.y, -l.z]
                                    for l in world_landmarks])
            image_points = np.float32(
            [[l.x * self.image_width, l.y * self.image_height] for l in hand_landmarks])
        
            success, rvecs, tvecs, = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.distortion,
                flags=cv2.SOLVEPNP_SQPNP
            )

            # Attempt to get the angles. Did not work properly
            # rmat, jac = cv2.Rodrigues(rvecs)
            # quat = R.from_matrix(rmat).as_quat()
            # print(R.from_matrix(rmat).as_euler('zxy', degrees=True))
            # angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # print(np.array(angles))
            # quat_total.append(quat)
            
            # needs to 4x4 because you have to use homogeneous coordinates
            transformation = np.eye(4)
            transformation[0:3, 3] = tvecs.squeeze()
            # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

            # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate(
                (model_points, np.ones((21, 1))), axis=1)

            # apply the transformation
            world_points = -model_points_hom.dot(np.linalg.inv(transformation).T) # Matthias added this minus to be in the optical_frame tf frame of ROS
            world_points_total.append(world_points)
            # Get the landmarks for the index finger and thumb
            # index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

            # # Calculate the rotation angle between the index finger and thumb
            # angle = np.arctan2(
            #     index_finger.y - thumb.y, index_finger.x - thumb.x
            # )

            if world_points_total:
                msg = self.generate_leap_msg(world_points_total, quat_total)
                self.pub_ros.publish(msg)

        # # Flip the image horizontally for a selfie-view display.
        # # print(world_points_total)
        output_image_bgr = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
        self.annotated_img = draw_landmarks_on_image(output_image_bgr, result)
        # cv2.imshow('MediaPipe Hands', cv2.flip(annotated_image[:, :, ...], 1))
        # #cv2.imwrite("MediaPipeHands.png", annotated_image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     rospy.is_shutdown = True


    def generate_leap_msg(self, world_points_total, quat_total):
        msg = leapros()


        # Attempt to get the rotation
        # # Define 3 points for hand plane
        # plane_points = np.asarray([world_points_total[0][LANDMARK_INDEX_DICT["WRIST"]][0:3], 
        #                            world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_MCP"]][0:3], 
        #                            world_points_total[0][LANDMARK_INDEX_DICT["PINKY_MCP"]][0:3]])
        # # Get normal vector to hand plane and normalize it
        # normal_vector = np.cross(plane_points[2] - plane_points[0], plane_points[1] - plane_points[2])
        # normal_vector /= np.linalg.norm(normal_vector)

        # # Compute the rotation axis and angle
        # angle = np.arccos(normal_vector[2])
        # axis = np.cross(np.array([0, 0, 1]), normal_vector)
        # axis = axis / np.linalg.norm(axis)
        
        # # Convert the axis-angle representation to a quaternion
        # quat = R.from_rotvec(angle * axis).as_quat()


        index_finger_tip_noisy = np.asarray(world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][0:3])
        index_finger_mcp_noisy = np.asarray(world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_MCP"]][0:3])

        thumb_tip_noisy  = np.asarray(world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][0:3])
        thumb_mcp_noisy  = np.asarray(world_points_total[0][LANDMARK_INDEX_DICT["THUMB_MCP"]][0:3])
        wrist_noisy  = np.asarray(world_points_total[0][LANDMARK_INDEX_DICT["WRIST"]][0:3])


        if self.filter_index_tip is None:
            min_cutoff = 1.0
            beta = 0.0004
            print("initialize")
            self.filter_index_tip = OneEuroFilter(index_finger_tip_noisy , min_cutoff = min_cutoff, beta = beta)
            self.filter_index_mcp = OneEuroFilter(index_finger_mcp_noisy , min_cutoff = min_cutoff, beta = beta)
            self.filter_thumb_tip = OneEuroFilter(thumb_tip_noisy , min_cutoff = min_cutoff, beta = beta)
            self.filter_thumb_mcp = OneEuroFilter(thumb_mcp_noisy , min_cutoff = min_cutoff, beta = beta)
            self.filter_wrist = OneEuroFilter(wrist_noisy , min_cutoff = min_cutoff, beta = beta)

        index_finger_tip = self.filter_index_tip(index_finger_tip_noisy)
        index_finger_mcp = self.filter_index_mcp(index_finger_mcp_noisy)
        thumb_tip = self.filter_thumb_tip(thumb_tip_noisy)
        thumb_mcp = self.filter_thumb_mcp(thumb_mcp_noisy)
        wrist = self.filter_wrist(thumb_mcp_noisy)

        # FOR DEBUGGING:
        # index_finger_tip = np.asarray([0.,0.,0.9])
        # index_finger_mcp = np.asarray([-0.1,0.,0.9])

        # thumb_tip = np.asarray([0.1,0.1,0.8])
        # thumb_mcp = np.asarray([-0.05,0.1,0.8])


        tip_dist = index_finger_tip - thumb_tip
        mcp_dist = index_finger_mcp - thumb_mcp
        middle_point_tips = thumb_tip + tip_dist/2
        middle_point_mcp = thumb_mcp + mcp_dist/2
        
        # Old Transformation using the middle point of mcp and tips
        # new_x = tip_dist/np.linalg.norm(tip_dist)

        # new_y = (middle_point_tips-middle_point_mcp)-(np.dot((middle_point_tips-middle_point_mcp),new_x)*new_x)
        # new_y = new_y/np.linalg.norm(new_y)

        # new_z = np.cross(new_x, new_y)
        # new_z = new_z/np.linalg.norm(new_z)

        # rot_mat = np.column_stack((new_x, new_y, new_z))

        # new_x = tip_dist/np.linalg.norm(tip_dist)

        #new_y = (middle_point_tips-middle_point_mcp)-(np.dot((middle_point_tips-middle_point_mcp),new_x)*new_x)
        new_y = middle_point_tips - wrist
        new_y = new_y/np.linalg.norm(new_y)

        new_x = (index_finger_tip-thumb_tip)-(np.dot((index_finger_tip-thumb_tip),new_y)*new_y)
        new_x = new_x/np.linalg.norm(new_x)

        new_z = np.cross(new_x, new_y)
        new_z = new_z/np.linalg.norm(new_z)

        rot_mat = np.column_stack((new_x, new_y, new_z))

        #print(np.linalg.det(rot_mat))

        #print("X:", new_x, "Y:", new_y, "Prod:", np.dot(new_x, new_y) )
        # print(new_x)
        quat = R.from_matrix(rot_mat).as_quat()


        

        

        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "oak_rgb_camera_optical_frame"
        t.child_frame_id = "middle_point"
        t.transform.translation.x = middle_point_tips[0]
        t.transform.translation.y = middle_point_tips[1]
        t.transform.translation.z = middle_point_tips[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]


        self.br.sendTransform(t)



        ############# MARKERS FOR DEBUGGING ############
        # Create a single marker for the sphere list
        marker = Marker()
        marker.header.frame_id = "oak_rgb_camera_optical_frame"
        marker.type = Marker.SPHERE_LIST
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        # Define the points for the sphere list
        marker.points.append(Point(thumb_tip[0], thumb_tip[1], thumb_tip[2]))
        marker.points.append(Point(index_finger_tip[0], index_finger_tip[1], index_finger_tip[2]))
        marker.points.append(Point(middle_point_tips[0], middle_point_tips[1], middle_point_tips[2]))
        marker.points.append(Point(middle_point_mcp[0], middle_point_mcp[1], middle_point_mcp[2]))

        # Define the colors for each point
        marker.colors.append(ColorRGBA(1.0, 0.0, 0.0, 1.0))  # Red
        marker.colors.append(ColorRGBA(0.0, 1.0, 0.0, 1.0))  # Green
        marker.colors.append(ColorRGBA(0.0, 0.0, 1.0, 1.0))  # Blue
        marker.colors.append(ColorRGBA(1.0, 1.0, 0.0, 1.0))  # Yellow


        # Create an arrow marker pointing from the first sphere to the second sphere
        arrow = Marker()
        arrow.header.frame_id = "oak_rgb_camera_optical_frame"
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.01  # Arrow length
        arrow.scale.y = 0.015  # Arrow width
        arrow.scale.z = 0.02  # Arrow height
        arrow.color = ColorRGBA(1.0, 1.0, 0.0, 1.0)  # Yellow

        arrow.points.append(marker.points[3])  # Start point of the arrow
        arrow.points.append(marker.points[2])  # End point of the arrow



        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        self.marker_publisher.publish(marker)

        # arrow.header.stamp = rospy.Time.now()
        # arrow.action = Marker.ADD
        # self.marker_publisher.publish(arrow)


        # Use quat we get from PnP
        #quat = quat_total[0]
        #print(quat)

        msg.palmquat.x = quat[0]
        msg.palmquat.y = quat[1]
        msg.palmquat.z = quat[2]
        msg.palmquat.w = quat[3]
        #print(msg.palmquat)

        distance_thumb_index = np.linalg.norm(world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]]-world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]])

        msg.pinch_strength.data = float(np.clip(1.-distance_thumb_index/0.1, 0., 1.))

        msg.palmpos.x = world_points_total[0][LANDMARK_INDEX_DICT["WRIST"]][0]
        msg.palmpos.y = world_points_total[0][LANDMARK_INDEX_DICT["WRIST"]][1]
        msg.palmpos.z = world_points_total[0][LANDMARK_INDEX_DICT["WRIST"]][2]

        #print(msg.palmpos)

        # THUMB
        msg.thumb_metacarpal.x = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_CMC"]][0]
        msg.thumb_metacarpal.y = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_CMC"]][1]
        msg.thumb_metacarpal.z = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_CMC"]][2]

        msg.thumb_proximal.x = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_MCP"]][0]
        msg.thumb_proximal.y = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_MCP"]][1]
        msg.thumb_proximal.z = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_MCP"]][2]

        msg.thumb_intermediate.x = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_IP"]][0]
        msg.thumb_intermediate.y = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_IP"]][1]
        msg.thumb_intermediate.z = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_IP"]][2]

        msg.thumb_distal.x = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][0]
        msg.thumb_distal.y = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][1]
        msg.thumb_distal.z = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][2]

        msg.thumb_tip.x = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][0]
        msg.thumb_tip.y = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][1]
        msg.thumb_tip.z = world_points_total[0][LANDMARK_INDEX_DICT["THUMB_TIP"]][2]

        # INDEX FINGER
        msg.index_metacarpal.x = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_MCP"]][0]
        msg.index_metacarpal.y = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_MCP"]][1]
        msg.index_metacarpal.z = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_MCP"]][2]

        msg.index_proximal.x = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_PIP"]][0]
        msg.index_proximal.y = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_PIP"]][1]
        msg.index_proximal.z = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_PIP"]][2]

        msg.index_intermediate.x = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_DIP"]][0]
        msg.index_intermediate.y = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_DIP"]][1]
        msg.index_intermediate.z = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_DIP"]][2]

        msg.index_distal.x = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][0]
        msg.index_distal.y = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][1]
        msg.index_distal.z = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][2]

        msg.index_tip.x = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][0]
        msg.index_tip.y = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][1]
        msg.index_tip.z = world_points_total[0][LANDMARK_INDEX_DICT["INDEX_FINGER_TIP"]][2]

        # MIDDLE FINGER
        msg.middle_metacarpal.x = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_MCP"]][0]
        msg.middle_metacarpal.y = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_MCP"]][1]
        msg.middle_metacarpal.z = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_MCP"]][2]

        msg.middle_proximal.x = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_PIP"]][0]
        msg.middle_proximal.y = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_PIP"]][1]
        msg.middle_proximal.z = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_PIP"]][2]

        msg.middle_intermediate.x = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_DIP"]][0]
        msg.middle_intermediate.y = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_DIP"]][1]
        msg.middle_intermediate.z = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_DIP"]][2]

        msg.middle_distal.x = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][0]
        msg.middle_distal.y = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][1]
        msg.middle_distal.z = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][2]

        msg.middle_tip.x = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][0]
        msg.middle_tip.y = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][1]
        msg.middle_tip.z = world_points_total[0][LANDMARK_INDEX_DICT["MIDDLE_FINGER_TIP"]][2]

        # RING FINGER
        msg.ring_metacarpal.x = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_MCP"]][0]
        msg.ring_metacarpal.y = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_MCP"]][1]
        msg.ring_metacarpal.z = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_MCP"]][2]

        msg.ring_proximal.x = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_PIP"]][0]
        msg.ring_proximal.y = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_PIP"]][1]
        msg.ring_proximal.z = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_PIP"]][2]

        msg.ring_intermediate.x = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_DIP"]][0]
        msg.ring_intermediate.y = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_DIP"]][1]
        msg.ring_intermediate.z = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_DIP"]][2]

        msg.ring_distal.x = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][0]
        msg.ring_distal.y = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][1]
        msg.ring_distal.z = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][2]

        msg.ring_tip.x = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][0]
        msg.ring_tip.y = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][1]
        msg.ring_tip.z = world_points_total[0][LANDMARK_INDEX_DICT["RING_FINGER_TIP"]][2]

        # PINKY
        msg.pinky_metacarpal.x = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_MCP"]][0]
        msg.pinky_metacarpal.y = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_MCP"]][1]
        msg.pinky_metacarpal.z = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_MCP"]][2]

        msg.pinky_proximal.x = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_PIP"]][0]
        msg.pinky_proximal.y = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_PIP"]][1]
        msg.pinky_proximal.z = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_PIP"]][2]

        msg.pinky_intermediate.x = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_DIP"]][0]
        msg.pinky_intermediate.y = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_DIP"]][1]
        msg.pinky_intermediate.z = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_DIP"]][2]

        msg.pinky_distal.x = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][0]
        msg.pinky_distal.y = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][1]
        msg.pinky_distal.z = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][2]

        msg.pinky_tip.x = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][0]
        msg.pinky_tip.y = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][1]
        msg.pinky_tip.z = world_points_total[0][LANDMARK_INDEX_DICT["PINKY_TIP"]][2]

        return msg
    
    def list_ports(self):
        is_working = True
        dev_port = 0
        working_ports = []
        available_ports = []
        while is_working:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                is_working = False
                print("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    print("Port %s is working and reads images (%s x %s)" %
                        (dev_port, h, w))
                    working_ports.append(dev_port)
                else:
                    print("Port %s for camera ( %s x %s) is present but does not reads." % (
                        dev_port, h, w))
                    available_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports






if __name__ == '__main__':

    rospy.init_node('hsr_node', anonymous=True)

    media_pipe_hands = MediaPipeHands(camera_name="/oak/rgb")

    rospy.spin()

    #media_pipe_hands.detection_loop()

    #media_pipe_hands.cap.release()
