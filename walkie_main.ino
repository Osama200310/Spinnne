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

const int STEER_CORRECTION = 114;
const int STEERING_FACTOR = 100;
const int MIN_THRESHHOLD = 40;
const int MAX_THRESHHOLD = 80;
const int CRIT_THRESHHOLD = 20;
const int TURNING_RATE = 1000;
const int MOVING_RATE = 1000;
const int BACKOFF_RATE = 1500;
const int DANCE_FACTOR = 80;
const int DANCE_DELAY = 500;
const int MVMNT_BUFFER = 500;
const int DIST_OFFSET = 8;

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
                MSound(1, 100, 1000);
                ROBOT_INIT();                   // reset etc.
                ROBOT_PWR_ON();                 // power on
                ROBOT_HEIGHT(_hight);           // init hight
                ROBOT_SPEED(10);                // init speed (value 10 is fast and value 200 is very slow)
                ROBOT_GAINT_MODE(WAVE_24);
                delay(500);

                bool is_dancing = true;
                ROBOT_ROTATE_MODE(128,128,128,128);
                delay(DANCE_DELAY);

                // dancing loop
                while(is_dancing == true) {

                    ROBOT_ROTATE_MODE(128 + DANCE_FACTOR,128,128,128 - DANCE_FACTOR / 2);
                    delay(DANCE_DELAY);

                    ROBOT_ROTATE_MODE(128,128,128,128 + DANCE_FACTOR / 2);
                    delay(DANCE_DELAY);

                    ROBOT_ROTATE_MODE(128 - DANCE_FACTOR,128,128,128 - DANCE_FACTOR / 2);
                    delay(DANCE_DELAY);

                    ROBOT_ROTATE_MODE(128,128,128,128 + DANCE_FACTOR / 2);
                    delay(DANCE_DELAY);

                    if (!digitalRead(T2)) {
                        delay(50);
                        if(!digitalRead(T2)) {
                            is_dancing = false;
                        }
                    }
                }

                // steering loop / steering mode
                bool is_close = false;
                double detected_dist = 0;
                bool change_dir = false;
                bool is_running = true;
                bool backoff = false;

                while (is_running == true) {
                    distance_cm = mySensor.distance();

                    if (distance_cm < MIN_THRESHHOLD && is_close == false) {
                        detected_dist = distance_cm;
                        is_close = true;   
                    } else if (distance_cm > MAX_THRESHHOLD) {
                        is_close = false;
                        change_dir = false;
                        backoff = false;
                    }

                    if (is_close == true) {
                        if (distance_cm < (detected_dist - DIST_OFFSET) && change_dir == false) {
                            change_dir = true;
                        }
                        else if (distance_cm < CRIT_THRESHHOLD && backoff == false) {
                            backoff = true;
                        }
                    }



                    if (is_close == true) {
                        if (change_dir == false)
                            ROBOT_MOVE(128, STEER_CORRECTION + STEERING_FACTOR, 128);
                        else
                            ROBOT_MOVE(128, STEER_CORRECTION - STEERING_FACTOR, 128);
                        } 
                    else if (backoff == true)
                        ROBOT_MOVE(128, STEER_CORRECTION, 255);
                    else
                        ROBOT_MOVE(128, STEER_CORRECTION, 0);

                    delay(MVMNT_BUFFER);

                    
                    if (Serial.available() > 0) {
                        int input = Serial.read() - '0';
                        if (input == 1) {
                            is_running = false;
                        }
                    }
                    
                }

                
                ROBOT_MOVE(128,128,128);        // stop
                delay(500);
                
                ROBOT_HEIGHT(0);                // sit down
                delay(1000);  

                for (int i = 0; i < 10; i++)      // end beeping
                {
                    MSound(3, 100, 1000, 50, 2000, 100, 3500);
                    delay(500);
                }

                delay(5000);
                
                ROBOT_PWR_OFF();                // power off
                delay(1500);                
            }
        }
    }
}

