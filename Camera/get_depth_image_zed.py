import pyzed.sl as sl
import cv2 
import numpy as np


def main():
    # Create a Camera object
    zed = sl.Camera()
    max_depth = 1500
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Set the depth mode to performance (fastest)
    init_params.depth_minimum_distance = 300
    init_params.depth_maximum_distance = max_depth
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
    
    # Open the camera
    err = zed.open(init_params)
    #err = zed.open()
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))
    

    # Capture 50 frames and stop
    i = 0
    image_zed = sl.Mat()
    image_depth_zed = sl.Mat()
    depth_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    while i < 10:
        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Grab the zed images
            # A new image is available if grab() returns sl.ERROR_CODE.SUCCESS
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
            zed.retrieve_image(image_depth_zed, sl.VIEW.DEPTH) # Get the depth image
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH) # Retrieve depth matrix. Depth is aligned on the left RGB image
            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud
            i = i+1

            # Convert to numpy/cv2 arrays
            # Use get_data() to get the numpy array of the left image
            image_cv2 = image_zed.get_data()
            # numpy array of the rgbdepth image
            image_rgbdepth = image_depth_zed.get_data()
            # numpy array of the depth data
            depth_image = depth_zed.get_data()
            #print(depth_image)
            depth_image = np.where(np.isnan(depth_image), max_depth, depth_image)
            depth_image = np.where(np.isneginf(depth_image), max_depth, depth_image)
            depth_image = np.where(np.isinf(depth_image), max_depth, depth_image)
            print(depth_image)

            img = np.array(depth_image/max_depth*255, dtype=np.uint8)
            print(img)

            # Display images
            cv2.imshow('depth image', img) 
            key = 0xFF & cv2.waitKey(0)
            if key == ord('q'):
                break

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()