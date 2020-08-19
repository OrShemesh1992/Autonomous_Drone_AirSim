
import airsim
import numpy as np
import cv2
import math
import time
#set segmentation colors
def set_segmentation_colors():
    found = client.simSetSegmentationObjectID("[\w]*", 0, True)

    found = client.simSetSegmentationObjectID('Landscape_1', 0)

    found = client.simSetSegmentationObjectID('InstancedFoliageActor_0', 1)


# this function calculate the disparity image, write it and return it
def calc_disparity_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DisparityNormalized, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # original image is fliped vertically
    img2d = np.flipud(img2d)
    return img2d


# this function calculate the depth planner image, write it and return it
def calc_depth_planner_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # original image is fliped vertically
    img2d = np.flipud(img2d)
    return img2d


# this function calculate the depth perspective image, write it and return it
def calc_depth_perspective_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # original image is fliped vertically
    img2d = np.flipud(img2d)
    return img2d


# this function calculate the depth vis image, write it and return it
def calc_depth_vis_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])
    response = responses[0]
    img2d = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    # original image is fliped vertically
    img2d = np.flipud(img2d)
    return img2d


# this function calculate the segmentation image, write it and return it
def calc_segmentation_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_rgb = img1d.reshape((response.height, response.width, 3))  # reshape array to 3 channel image array H X W X 3
    return img_rgb

def image_proc_algorithm():
    depth_planner_image = calc_depth_planner_image()
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
    # Thickening parts of pixels in the image with some kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    erode_image = cv2.erode(out_, kernel)

    # Make border to help to Canny algorithm to surround the whole edges of object
    erode_image = cv2.copyMakeBorder(erode_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)

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

    # Iterate over all contours and draw all box
    depth = float('inf')
    for i, c in enumerate(contours):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        pos1, pos2, pos3, pos4 = box
        area = abs((pos1[0]*pos2[1]-pos1[1]*pos2[0])+(pos2[0]*pos3[1]-pos2[1]*pos3[0]) +
                   (pos3[0]*pos4[1]-pos3[1]*pos4[0])+(pos4[0]*pos1[1]-pos4[1]*pos1[0]))/2
        if area > 1000:
            min_x, max_x = (min([pos1[0], pos2[0], pos3[0], pos4[0]]), max([pos1[0], pos2[0], pos3[0], pos4[0]]))
            min_y, max_y = (min([pos1[1], pos2[1], pos3[1], pos4[1]]), max([pos1[1], pos2[1], pos3[1], pos4[1]]))
            if max_x >= np.shape(out_)[1]:
                max_x = np.shape(out_)[1]
            if max_y >= np.shape(out_)[0]:
                max_y = np.shape(out_)[0]
            depth = float('inf')
            for k in range(min_y, max_y):
                for j in range(min_x, max_x):
                    if out_[k][j] == 0.0:
                        if depth_planner_image[k, j] < depth:
                            depth = depth_planner_image[k, j]
                        break
            cv2.drawContours(main, [box], 0, (0, 0, 255), 2)

    return depth

def Escape_algorithm():
    # connect to the AirSim simulator
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    set_segmentation_colors()
    client.takeoffAsync().join()

    # z of -20 is 20 meters above the original launch point.
    z = -20

    # Fly given velocity vector for 5 seconds
    duration = 5
    speed  =1

    vx = speed
    vy = 0
    dgree=0
    count = 0
    point=0
    move=True
    while True:
        if count == 2:
            count = 0
        result = image_proc_algorithm()
        print(result)
        if result < 5:
            move=True
            if point==0:
                if count == 0:
                    vx = 0
                    vy = -speed
                    # turn left
                    print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=270")
                    client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 270))
                elif count == 1:
                    vx = -speed
                    vy = 0
                    # straight
                    print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=180")
                    client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 180))
                time.sleep(1)
            elif point==1:
                z = -10
                if count == 0:
                    vx = 0
                    vy = -speed
                    # turn left
                    print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=270")
                    client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 270))
                elif count == 1:
                    vx = speed
                    vy = 0
                    # straight
                    print("moving by velocity vx=" + str(vx) + ", vy=" + str(vy) + ", yaw=0")
                    client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                airsim.YawMode(False, 0))
                time.sleep(1)
            count += 1
        else:
            count=0
            x = client.getMultirotorState().kinematics_estimated.position.x_val
            y = client.getMultirotorState().kinematics_estimated.position.y_val
            print(x,y)
            if -108<float(x) and float(x)<-102 and -83<float(y) and float(y) <-77:
                point=1
                move = True
            elif 99>float(x) >94 and  -107<float(y) <-100:
                client.landAsync().join()
                client.armDisarm(False)
                client.reset()
                # let's quit cleanly
                client.enableApiControl(False)
            if move==True:
                move=False
                if point ==0:
                    print("point 1")
                    x1= -105.86600494384766
                    y1=-79.34103393554688
                    dgree = 180+math.degrees(math.atan((y - y1) / (x - x1)))
                    client.moveToPositionAsync(x1, y1, z, speed, yaw_mode=airsim.YawMode(False, dgree))
                elif point==1:
                    print("point 2")
                    x1=  95.15351867675781
                    y1= -104.22108459472656
                    dgree = math.degrees(math.atan((y - y1) / (x - x1)))
                    client.moveToPositionAsync(x1, y1, z, speed ,yaw_mode=airsim.YawMode(False, dgree))
# main
if __name__ == "__main__":
    client = airsim.MultirotorClient()
    Escape_algorithm()