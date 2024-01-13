import cv2
import numpy as np
import os, shutil
import pyzed.sl as sl

image_path = '/home/stefan/Documents/Master/MV_Project/V2V-Pytorch/V2V-PoseNet-PyTorch/pose_estimation/img3/'


def save_images(bgr_l, bgr_r, depth, i):
    np.save(image_path + "depth/" + str(i), depth)
    np.save(image_path + "rgb/left/" + str(i), bgr_l)
    np.save(image_path + "rgb/right/" + str(i), bgr_r)


if __name__ == "__main__":
    min_depth = 300
    max_depth = 1500
    
    # delte images and build folders
    shutil.rmtree(image_path)
    
    os.mkdir(image_path)
    os.mkdir(image_path + 'depth')
    os.mkdir(image_path + 'rgb')
    os.mkdir(image_path + 'rgb/left')
    os.mkdir(image_path + 'rgb/right')
    
    # camera
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
    init_params.depth_minimum_distance = min_depth
    init_params.depth_maximum_distance = max_depth
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

    err = zed.open(init_params)
    
    # return false if open was not successful
    if err != sl.ERROR_CODE.SUCCESS:
        exit()
    
    # print serial number (for checking)
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))


    i = 0

    while i < 200:

        image_zed = sl.Mat()
        image_zed_r = sl.Mat()
        depth_zed = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
            zed.retrieve_image(image_zed_r, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH) # Retrieve depth matrix. Depth is aligned on the left RGB image
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) 
            
            # Use get_data() to get the numpy array
            image_bgra = image_zed.get_data()
            image_bgra_r = image_zed_r.get_data()
            depth_image = depth_zed.get_data()

            # Transform color image
            image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR) 
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
            image_bgr_r = cv2.cvtColor(image_bgra_r, cv2.COLOR_BGRA2BGR) 
            image_rgb_r = cv2.cvtColor(image_bgr_r, cv2.COLOR_BGR2RGB) 

            # Adjust depth image
            # Filter out nan, -inf, inf values
            depth_image = np.where(np.isnan(depth_image), max_depth, depth_image)
            depth_image = np.where(np.isneginf(depth_image), min_depth, depth_image)
            depth_image = np.where(np.isinf(depth_image), max_depth, depth_image)
            
            save_images(image_rgb, image_rgb_r, depth_image, i)

            img = np.array(depth_image/np.max(depth_image)*255, dtype=np.uint8)
            cv2.imshow("img", img)
            cv2.waitKey(5)
            

        i = i+1