/******************************************************************************
Created with PROGRAMINO IDE for Arduino
Project     : Moving_03.ino
Libraries   : SoftwareSerial.h, Hexapod_Lib.h
Author      : UlliS
******************************************************************************

[ INFO: This example is for ARDUINO UNO or compatible boards and NodeMCU ]

- The sample show the basic movements with the hexapod library
- Main focus is the ROBOT_MOVE() function

******************************************************************************/

// Arduino or NodeMCU (select one)
#define ARDUINO
//#define NODEMCU

#include <Hexapod_Lib.h>
#include "SharpIR.h"

#define IR_PIN A5
#define MODEL 1080

const int STEER_CORRECTION = 117;
const int STEERING_FACTOR = 100;
const int MIN_THRESHHOLD = 40;
const int MAX_THRESHHOLD = 60;
const int CRIT_THRESHHOLD = 20;
const int TURNING_RATE = 1000;
const int MOVING_RATE = 1000;
const int BACKOFF_RATE = 2000;

int distance_cm = 0;

SharpIR mySensor = SharpIR(IR_PIN, MODEL);

/******************************************************************************
INIT
******************************************************************************/
void setup() 
{
    // high-Z for the audio output
    pinMode(PA_PIN,INPUT);
    digitalWrite(PA_PIN,LOW);
    
    // switches T1 and T2
    #ifdef ARDUINO
        pinMode(T1,INPUT);
        pinMode(T2,INPUT);
        //digitalWrite(T1,HIGH); // use internal pull-up resistor
        //digitalWrite(T2,HIGH); // use internal pull-up resistor
    #endif   
    
    // open serial communications and wait for port to open:
    Serial.begin(SERIAL_STD_BAUD);
    while(!Serial) 
    {
        ;  // wait for serial port to connect. Needed for native USB port only
    }
    
    // set the data rate for the SoftwareSerial port (User-Board to Locomotion-Controller)
    SERIAL_CMD.begin(SERIAL_CMD_BAUD);
    
    // reset the Locomotion-Controller
    ROBOT_RESET();
    delay(250);
    ROBOT_RESET();
    delay(150);
    ROBOT_RESET();
    
    // wait for Boot-Up
    delay(1500);
    ROBOT_INIT();
    
    // print a hello world over the USB connection
    Serial.println("> Hello here is the C-Control Hexapod!");
}

/******************************************************************************
MAIN
******************************************************************************/
void loop() 
{
    // main loop
    while(1)
    {
        
        int _hight = 60;    // init robot hight
        
        if(!digitalRead(T1)) 
        {
            delay(50);
            if(!digitalRead(T1))
            {   
                ROBOT_INIT();                   // reset etc.
                ROBOT_PWR_ON();                 // power on
                ROBOT_HEIGHT(_hight);           // init hight
                ROBOT_SPEED(10);                // init speed (value 10 is fast and value 200 is very slow)
                ROBOT_GAINT_MODE(WAVE_24);
                delay(500);

                // steering loop / steering mode
                bool is_close = false;
                double detected_dist = 0;
                bool change_dir = false;
                bool is_running = true;

                while(is_running) {
                    distance_cm = mySensor.distance();

                    if (distance_cm < MIN_THRESHHOLD) {
                        if (distance_cm < CRIT_THRESHHOLD) {
                            ROBOT_MOVE(128, STEER_CORRECTION, 255);
                            delay(BACKOFF_RATE);
                        }                
                        if (is_close == false) {
                            detected_dist = distance_cm;             
                            is_close = true;
                        }   
                    } else if (is_close == true && distance_cm > MAX_THRESHHOLD){
                        is_close = false;
                        change_dir = false;
                    }

                    if (is_close == true) {
                        if (!change_dir == false || distance_cm < detected_dist) {
                            change_dir = true;
                            ROBOT_MOVE(128, STEER_CORRECTION - STEERING_FACTOR, 128);
                            delay(TURNING_RATE);
                        } else {
                            ROBOT_MOVE(128, STEER_CORRECTION + STEERING_FACTOR, 128);
                            delay(TURNING_RATE);
                        }
                    } else {
                        ROBOT_MOVE(128, STEER_CORRECTION, 0);
                        delay(MOVING_RATE);
                    }

                    if (!digitalRead(T2)) {
                        delay(50);
                        if(!digitalRead(T2)) {
                            is_running = false;
                        }
                    }
                }
                
                ROBOT_MOVE(128,128,128);        // stop
                delay(500);
                
                ROBOT_HEIGHT(0);                // sit down
                delay(1000);                
                
                ROBOT_PWR_OFF();                // power off
                delay(1500);                
            }
        }
    }
}

