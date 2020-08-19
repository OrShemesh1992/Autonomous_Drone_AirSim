# Autonomous Drone AirSim
![image](https://user-images.githubusercontent.com/44946807/90451881-84edae80-e0f5-11ea-9e40-d4ef2afff62b.png)


## Authors
 [Or Shemesh](https://github.com/OrShemesh1992)
 
 [Maxim Marmer](https://github.com/MarmerMax)
 
 [Netanel Davidov](https://github.com/netanel208)



## About the project:

### Goals and objectives:
 The goal of the project is to build a simulation and algorithm that will make the drone fly and dodge obstacles autonomously.
 The drone takes off and flies in a pre-defined trajectory where it must reach certain destinations. But he must consider the environment given to him i.e.
 avoid collisions with objects and crashes.
 Drone requirements that the code relies on are: GPS, camera, LIDARS control components that the drone should have, distance sensors, altitude, motors, etc.
 We aim to use mainly a camera that will allow the drone to detect obstacles that are about to approach its path and has a high potential to collide with them (we will define
 what is the same potential) by algorithms in computer vision and image processing, and distance sensors to give appropriate commands to control its movement. The algorithem
 will use the camera to try to calculate the extent of the obstacle and thus try to get around it and complete the route, with as few deviations from the route that the
 drone needs to complete.

### Development stages:
- Familiarity with the `Python` libraries:
  - `airsim`
  - `cv2`
  - `numpy`
  - `os`
- Finding a desired route / routes that we would like a drone to fly on.
- Investigating and developing an algorithm that uses the camera in the drone while modeling the objects in the image that the drone is supposed to evade. 
  Search for advanced computer vision and image processing algorithms that analyze video in real time and use it to identify objects, trial and error.
- Converting the information obtained from the object identification in the image to useful information that the drone will use to calculate its distance
  from the object with the greatest potential to collide with it.
- The evasion algorithm - takes into account the information obtained from the analysis of the current image. In a situation where there is a potential for a collision, the
  algorithm will evade the nearest obstacle and return to the orbit in order to reach the target point defined for it.
  
### Algorithms:
#### Computer Vision:
- Every second the algorithm takes a frame / image from the camera on the drone from several necessary channels:
  - `Depth planner image`
  - `Depth vis image`
  - `Segmentation image`
- For starters the algorithm uses the segmentation image along with the depth vis image to display in a binary image objects that are up to 20 meters away from the drone:

  ![blackwhitetest3](https://user-images.githubusercontent.com/44946807/90443175-75b23500-e0e4-11ea-8807-84862887c2e4.png) 
  
- Then because the drone in the simulation flies 1 meter above the ground, it can be removed from the image analysis stages and we get:

  ![before erods](https://user-images.githubusercontent.com/44946807/90443554-1e609480-e0e5-11ea-9cf4-0c78b182e53e.PNG)

- At this point the objects in the image are identified, so we decided to use the Canny Edges Detectios algorithm to find and delineate the object edges in the image obtained in
  the previous step first:

  ![image](https://user-images.githubusercontent.com/44946807/90445725-cb88dc00-e0e8-11ea-9a06-7374820a3153.png)

- Then to mark the edges we used the Contours Approximation algorithm which at the end returns `boxes` that surround the objects that have edges in some closed polygonal shape
  identified in the image:
  - **Experiment 1 (will not work)-**
  
  ![image](https://user-images.githubusercontent.com/44946807/90446733-8ebde480-e0ea-11ea-851d-fa0e947ed619.png)
  
  
  - **Experiment 2 (work)-**
  
     As you can see we did not get a good result (a total of 109 red boxes were found in the picture) due to the fact that the edges do not form a single polygonal shape without
     amputations in the middle so to handle this you can use two methods to achieve a unified polygonal shape:
     1. Noise filtering.
     2. Thickening of pixels.
     
     We chose the **second** option because filtering noise will cause necessary filtering of information that filtering can cause the drone to collide with the edges of the
     bushes / trees / branches which is something we would not want to happen. 
     Therefore to do this you need to correlate with kernel on all the relevant pixels in the image (black) to thicken them so here too we have chosen the most optimal option of
     kernel(11x11) together with an Erods Algorithm that helps to thicken pixels with binary values together with kernel smoothly:
    
     5x5 kernel -
    
     ![after erods](https://user-images.githubusercontent.com/44946807/90448817-d8103300-e0ee-11ea-8c01-44dd0cef9d98.PNG)  
    
     7x7 kernel -
    
     ![after erods(7x7)](https://user-images.githubusercontent.com/44946807/90448870-e8281280-e0ee-11ea-8d92-a97225020a1a.PNG) 
    
     11x11 kernel -
    
     ![image](https://user-images.githubusercontent.com/44946807/90448993-27eefa00-e0ef-11ea-8d20-b23fbabe6f06.png)

      Many times parts of the object may be cut on the sides of the image which can affect the object recognition process so we added padding(border) to the edges of the image 
      so that the Canny Algorithm would have a better result:
       
      ![image](https://user-images.githubusercontent.com/44946807/90449714-a8fac100-e0f0-11ea-9923-b4fb36f077b7.png)
     
      And when we now run the Contours Detection Algorithm along with the Red Box Finding Algorithm we get:
     
      ![image](https://user-images.githubusercontent.com/44946807/90450120-946af880-e0f1-11ea-9384-d9ca721c4db7.png)
     
      The result is - 2 red boxs! 
      
      It should be noted that even after the optimization there are still noises that have been diagnosed as boxes so to filter them we decided to refer only to boxes with a
      space above 1000 pixels (because it makes the most sense to notice them).
      
- We then treated each box as an object and calculated for each box the distance from the drone using the depth images. With the help of these calculations the algorithm the 
  object that has great potential to collide with the drone.

## How to use the project:

1. download "Forest" map  from: https://github.com/microsoft/AirSim/releases/tag/v.1.2.2
2. run Forest.exe 
3. python Drone_Auto.py

## Sources:

* https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
* https://www.microsoft.com/en-us/ai/ai-lab-airsim-drones#:~:text=AI%20AirSim%20Drones,or%20people%20in%20the%20images.
* https://www.youtube.com/watch?v=-WfTr1-OBGQ
