import pyzed.sl as sl
import cv2 


def main():
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
    

    # Capture 50 frames and stop
    i = 0
    image_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    while i < 10:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)  # Get the image timestamp
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image_zed.get_width(), image_zed.get_height(), timestamp.get_milliseconds()))
            i = i + 1
            # Displaying the image 
            # Use get_data() to get the numpy array, is in BGRA format!
            image_cv2 = image_zed.get_data()
            cv2.imshow('image', image_cv2) 
            key = 0xFF & cv2.waitKey(0)
            if key == ord('q'):
                break

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()