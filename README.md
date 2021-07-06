# Perception-and-Control


Usage Instructions
Step 1: Ctrl+Alt+T to open a new terminal and open a rosmaster by roscore in this
terminal.
Step 2: Start the camera node. Ctrl+Shift+T to open a new tab in the terminal in step one.
Start the camera node by roslaunch demo_pointcloud.launch .
Step 3: Start the IMU node. Ctrl+Shift+T to open a new tab in the terminal in step one.
Start the IMU node by roslaunch xsens_driver xsens_driver.launch
device:=/dev/ttyTHS1 .
Step 4: Start the prediction node. Ctrl+Shift+T to open a new tab in the terminal in step
one. Start the prediction node by rosrun data_record_predict.py . 
The main script is data_record_predict.py .

![image](https://user-images.githubusercontent.com/86708111/124537546-ab6f2200-de4c-11eb-896b-bc4b86974029.png)
