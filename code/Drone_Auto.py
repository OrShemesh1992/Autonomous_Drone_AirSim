
import airsim
import numpy as np
import numpy
import os
import cv2
import time

def set_segmentation_colors():
    found = client.simSetSegmentationObjectID("[\w]*", 0, True)
    print("Set segmentation id to 0 to all objects done: %r" % found)

    found = client.simSetSegmentationObjectID('Landscape_1', 0)
    print("Set segmentation id to 255 to Landscape done: %r" % found)

    found = client.simSetSegmentationObjectID('InstancedFoliageActor_0', 1)
    print("Set segmentation id to 1 to Trees done: %r" % found)


# this function calculate the disparity image, write it and return it
def calc_disparity_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True, False)])
    response = responses[0]

    # img1d = np.array(response.image_data_float, dtype=np.float64)
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)


    # original image is fliped vertically
    img2d = np.flipud(img2d)

    # write image
    #airsim.write_pfm(os.path.normpath('images/test' + '.pfm'), img2d)
    return img2d


# this function calculate the depth planner image, write it and return it
def calc_depth_planner_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])
    response = responses[0]

    # img1d = np.array(response.image_data_float, dtype=np.float64)
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

    # original image is fliped vertically
    img2d = np.flipud(img2d)

    # write image
    # airsim.write_pfm(os.path.normpath('images/test1' + '.pfm'), img2d)
    return img2d


# this function calculate the depth perspective image, write it and return it
def calc_depth_perspective_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    response = responses[0]

    # img1d = np.array(response.image_data_float, dtype=np.float64)
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

    # original image is fliped vertically
    img2d = np.flipud(img2d)

    # write image
    # airsim.write_pfm(os.path.normpath('images/test2' + '.pfm'), img2d)
    return img2d


# this function calculate the depth vis image, write it and return it
def calc_depth_vis_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
    response = responses[0]

    # img1d = np.array(response.image_data_float, dtype=np.float64)
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

    # original image is fliped vertically
    img2d = np.flipud(img2d)

    # write image
    # airsim.write_pfm(os.path.normpath('images/test3' + '.pfm'), img2d)
    return img2d


# this function calculate the segmentation image, write it and return it
def calc_segmentation_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    response = responses[0]

    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgb = img1d.reshape((response.height, response.width, 3))  # reshape array to 3 channel image array H X W X 3

    # img_rgb = np.flipud(img_rgb)  # original image is fliped vertically

    # find unique colors
    # print(np.unique(img_rgb[:, :, 0], return_counts=True))  # red
    # print(np.unique(img_rgb[:, :, 1], return_counts=True))  # green
    # print(np.unique(img_rgb[:, :, 2], return_counts=True))  # blue

    # write image
    # cv2.imwrite(os.path.normpath('images/test3' + '.png'), img_rgb)  # write to png
    return img_rgb


def sortSecond(val):
    return val[1]


def image_proc_algorithm():
    # disparity_image = calc_disparity_image()
    depth_planner_image = calc_depth_planner_image()
    # depth_planner_image = np.flipud(depth_planner_image)  # original image is fliped vertically
    # depth_perspective_image = calc_depth_perspective_image()
    depth_vis_image = calc_depth_vis_image()
    depth_vis_image = np.flipud(depth_vis_image)  # original image is fliped vertically
    segmentation_image = calc_segmentation_image()

    # Algorithm
    # Part 1 - Create binary image only with trees in 100m view
    # Part 2 - Start Canny edge detection algorithm to emphsize them for objects detection algorithm
    # Part 3 - Find objects and draw rectangles

    # binary image with 1s values
    out_ = np.ones(depth_vis_image.shape, dtype=depth_vis_image.dtype)
    # Create binary image only with trees in 100m view
    for i in range(depth_vis_image.shape[0]):
        for j in range(depth_vis_image.shape[1]):
            if depth_vis_image[i][j] < 0.2:
                if np.array_equal(segmentation_image[i][j], np.array([42, 174, 203])):
                    out_[i][j] = 0.0
                else:
                    out_[i][j] = 1.0
            else:
                out_[i][j] = 1.0
    out_ = np.flipud(out_)  # original image is fliped vertically
    # airsim.write_pfm(os.path.normpath('images/tree in 100m' + '.pfm'), out_)

    # Thickening parts of pixels in the image with some kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    erode_image = cv2.erode(out_, kernel)

    # Make border to help to Canny algorithm to surround the whole edges of object
    erode_image = cv2.copyMakeBorder(erode_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    #airsim.write_pfm(os.path.normpath('images/blackwhitetest3' + '.pfm'), erode_image)

    # https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
    # Load images as greyscale but make main RGB so we can annotate in colour
    erode_image = np.flipud(erode_image)
    seg = np.array(erode_image).astype(dtype=np.uint8)
    seg = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX)
    main = np.copy(seg)
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)

    # Start Canny edge detection algorithm
    seg = cv2.Canny(seg, 0, 255)

    # Find external contours
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out_ = cv2.copyMakeBorder(out_, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    depth_planner_image = cv2.copyMakeBorder(depth_planner_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    # boxes = []
    depth = float('inf')
    # Iterate over all contours and draw all box
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        pos1, pos2, pos3, pos4 = box
        area = abs((pos1[0]*pos2[1]-pos1[1]*pos2[0])+(pos2[0]*pos3[1]-pos2[1]*pos3[0]) +
                   (pos3[0]*pos4[1]-pos3[1]*pos4[0])+(pos4[0]*pos1[1]-pos4[1]*pos1[0]))/2
        # print("box", box)
        # print("area = ", area)
        if area > 1000:
            # Find moment and centroid (Cx, Cy) for calculate the depth of tree in the box
            # m = cv2.moments(c)
            # cx = int(m['m10'] / m['m00'])
            # cy = int(m['m01'] / m['m00'])
            # print(cy, cx)
            min_x, max_x = (min([pos1[0], pos2[0], pos3[0], pos4[0]]), max([pos1[0], pos2[0], pos3[0], pos4[0]]))
            min_y, max_y = (min([pos1[1], pos2[1], pos3[1], pos4[1]]), max([pos1[1], pos2[1], pos3[1], pos4[1]]))
            # print(min_x, max_x, min_y, max_y)
            if max_x >= np.shape(out_)[1]:
                max_x = np.shape(out_)[1]
            if max_y >= np.shape(out_)[0]:
                max_y = np.shape(out_)[0]
            #b = False
            depth = float('inf')
            for k in range(min_y, max_y):
                for j in range(min_x, max_x):
                    if out_[k][j] == 0.0:
                        #b = True
                        # print("depth = ", depth_planner_image[k, j])
                        if depth_planner_image[k, j] < depth:
                            depth = depth_planner_image[k, j]
                        break
            # if b is not True:
            #     print("No depth found!")
            cv2.drawContours(main, [box], 0, (0, 0, 255), 2)
            # boxes.append((box, depth))

    # Show result and save
    # cv2.imwrite('images/result.png', main)

    # boxes.sort(key=sortSecond)
    return depth

    # Pass on erode_image and find the most close tree


# main
if __name__ == "__main__":
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    set_segmentation_colors()

    # print(client.getMultirotorState().kinematics_estimated.position)
    client.takeoffAsync().join()
    # AirSim uses NED coordinates so negative axis is up.
    # z of -20 is 20 meters above the original launch point.
    z = -20

    # Fly given velocity vector for 1 seconds
    duration = 1
    speed = 3

    vx = speed
    vy = 0

    count=0
    stop=0
    while True:
        stop+=1
        if stop == 50:
            break
        if count==4:
            count=0
        result=image_proc_algorithm()
        print(result)
        if  result<4:
            if count==0:
                vx = 0
                vy = -speed
                print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=270")
                client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 270))
            elif count==1:
                vx = -speed
                vy = 0
                print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=180")
                client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 180))
            elif count == 2:
                vx = 0
                vy = speed
                print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=90")
                client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 90))
            elif count == 3:
                vx = speed
                vy = 0
                print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=0")
                client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 0))
            count += 1
        else:
            print("moving by Position")
            client.moveToPositionAsync(-636.9027099609375,1013.6842041015625, -20, 2)

    airsim.wait_key('Press any key to reset to original state')
    client.armDisarm(False)
    client.reset()

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)

            # < Vector3r > {'x_val': -636.9027099609375,
            #               'y_val': -1013.6842041015625,
            #               'z_val': -18.373157501220703}
            # < Vector3r > {'x_val': -728.3554077148438,
            #               'y_val': -965.2791137695312,
            #               'z_val': -18.154285430908203}
            # < Vector3r > {'x_val': -1221.581298828125,
            #               'y_val': -590.2549438476562,
            #               'z_val': -18.378856658935547}
            # < Vector3r > {'x_val': 378.5397644042969,
            #               'y_val': 1.2118512131564785e-05,
            #               'z_val': -108.7598876953125}