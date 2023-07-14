# Water-Trash-Collector

the design and development of a mobile trash-collecting robot on a water surface based on YOLO v4 is a novel solution that has several advantages over traditional methods of waterway cleaning.

The robot is also highly maneuverable, allowing it to navigate through narrow waterways and collect trash from hard-to-reach areas. It can also be remotely controlled, making it easy to operate and monitor.

Overall, the use of YOLO v5 in the design and development of a mobile trash-collecting robot on the water surface is a novel solution that has the potential to revolutionize the way we clean and maintain our waterways. It offers a more efficient, cost-effective, and environmentally friendly approach to waterway cleaning and can help to address some of the most pressing environmental challenges of our time.


Prototype is Built on Raspberry PI with an Integrated Logitech camera for computer vision, Using low Level Object Detection Model (YOLO V4) for different classified objects to collect through the maneuver.

![image](https://github.com/annamalai0511/Water-Trash-Collector/assets/73933212/12648e4e-716a-499c-8cf5-eaa4e7e5ae4e)

![image](https://github.com/annamalai0511/Water-Trash-Collector/assets/73933212/15df4618-70b5-4f0f-8af6-bd6961d4895b)

![image](https://github.com/annamalai0511/Water-Trash-Collector/assets/73933212/6a6b86c0-c13d-44ba-acd1-afbc8f539d6a)

Video streamline captured by camera, is sent to rasp-pi for processing of video frames at 30 FPS rate
Rasp-pi then uses a pretrained model of trash on each frame to compute object detection in the background (It detects if there is any trash in that frame)
If it finds any, draws a bounding box around that detection on that particular frame.
Results then are sent to the moderator’s screen for further controlling in manual mode and in case of automation mode, processor computes it’s actions accordingly on it’s own.

[Screencast from 2023-07-14 23-36-02.webm](https://github.com/annamalai0511/Water-Trash-Collector/assets/73933212/9771c164-4f4e-457e-bbaf-e005fc048c39)
