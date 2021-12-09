# CSE_5349
Fall Detectiono Using Floor Sensor

To get sensor data, library for HX711 needs to be downloaded and stored in Arduino libraries. After that, open Arduino IDE, and the HX711_ADC can be found under examples section. Now, open Calibration sketch from HX711_ADC under examples section, load the sketch and program to the Arduino. At this point, the system is ready to communicate. To interact and get data from the system, simply using serial monitor in Arduino IDE or use a third partty terminal emulator such as Putty.
In our case, we use Putty to read the data because it is much more easy to read and log data from Putty. After logging data, we will clean the data using CleanData.py in Project DataCode folder. This program will output a new file only containning the data associated with the behavior. Then, simply copy this new file to CleanData_Code folder and run GUI.py. 
