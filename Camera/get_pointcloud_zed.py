import pyzed.sl as sl
import cv2 
import open3d as o3d
import numpy as np


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = 1
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA 
    init_params.depth_minimum_distance = 30
    init_params.depth_maximum_distance = 5000
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units
    
    # Open the camera
    err = zed.open(init_params)
    #err = zed.open()
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    zed_serial = zed.get_camera_information().serial_number
    print("Hello! This is my serial number: {0}".format(zed_serial))
    
    # Create an Open3D visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # Capture 50 frames and stop
    i = 0
    image_zed = sl.Mat()
    image_depth_zed = sl.Mat()
    depth_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    #runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
    while i < 1:
        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Grab the zed images
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Get the left image
            zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA) # Retrieve colored point cloud
            i = i+1

            # point3D = point_cloud.get_value(i, j)
            # x = point3D[0]
            # y = point3D[1]
            # z = point3D[2]
            # color = point3D[3]


            pointcloud_o3d = o3d.geometry.PointCloud()
            
            #points_zed = np.array(point_cloud_zed.get_data())
            xyz_zed = np.array(point_cloud_zed.get_data()[:, :, 0:3]).reshape(-1, 3)
            rgba_zed = np.ravel(point_cloud_zed.get_data()[:, :, 3]).view('uint8').reshape((-1, 4))
            print(xyz_zed)
            print(rgba_zed)
            # the color of open3d is in RGB space [0, 1]

            #points = np.zeros((points_zed.shape[0]*points_zed.shape[1], 3), dtype=np.float32)
            #points[:, 0] = points_zed[:, :, 0].flatten()
            #points[:, 1] = points_zed[:, :, 1].flatten()
            #points[:, 2] = points_zed[:, :, 2].flatten()
            #print("points",points)
            
            # Filter out naedn values
            mask_nan = np.isnan(xyz_zed)
            xyz_zed = xyz_zed[np.logical_not(mask_nan)].reshape(-1, 3)
            mask_nan_color = np.append(mask_nan, mask_nan[:, 2:3], axis=1)
            rgba_zed = rgba_zed[np.logical_not(mask_nan_color)].reshape(-1, 4)
            
            # Filter out -inf values
            mask_neginf = np.isneginf(xyz_zed)
            xyz_zed = xyz_zed[np.logical_not(mask_neginf)].reshape(-1, 3)
            mask_neginf_color = np.append(mask_neginf, mask_neginf[:, 2:3], axis=1)
            rgba_zed = rgba_zed[np.logical_not(mask_neginf_color)].reshape(-1, 4)

            # Filter out inf values
            mask_inf = np.isinf(xyz_zed)
            xyz_zed = xyz_zed[np.logical_not(mask_inf)].reshape(-1, 3)
            mask_inf_color = np.append(mask_inf, mask_inf[:, 2:3], axis=1)
            rgba_zed = rgba_zed[np.logical_not(mask_inf_color)].reshape(-1, 4)

            # Transform RGBa to RGB (in range [0, 1])
            rgb_zed = rgba_zed[:, 0:3] 
            alpha = rgba_zed[:, 3:4] / 255.0
            rgb_zed = rgb_zed * alpha + (1.0 - alpha) * 255.0
            rgb_o3d = rgb_zed/255.

            image_cv2 = image_zed.get_data()
            cv2.imshow('image', image_cv2) 
            key = 0xFF & cv2.waitKey(0)
            if key == ord('q'):
                break
            

            def rotate_view(vis):
                ctr = vis.get_view_control()
                ctr.rotate(2.0, 0.0)
                return False
    
            pointcloud_o3d.points = o3d.utility.Vector3dVector(xyz_zed)
            pointcloud_o3d.colors = o3d.utility.Vector3dVector(rgb_o3d)
            #vis.add_geometry(pointcloud_o3d)
            #vis.run()

            o3d.visualization.draw_geometries_with_animation_callback([pointcloud_o3d], rotate_view)
            #o3d.visualization.draw_geometries([pointcloud_o3d])            

    # Close the camera
    # vis.destroy_window()
    zed.close()


if __name__ == "__main__":
    main()