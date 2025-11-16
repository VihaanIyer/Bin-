/*
 * Arduino Servo Control Sketch
 * Receives commands from Python script via Serial
 * Format: "SERVO:pin:angle" (e.g., "SERVO:9:90")
 * 
 * Upload this sketch to your Arduino before running the Python script
 */

#include <Servo.h>

// Create servo objects for pins 9 and 10
Servo servo9;
Servo servo10;

String inputString = "";  // String to hold incoming data
boolean stringComplete = false;  // Whether the string is complete

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servos to pins
  servo9.attach(9);
  servo10.attach(10);
  
  // Set initial position to 0 degrees
  servo9.write(0);
  servo10.write(0);
  
  // Reserve space for input string
  inputString.reserve(200);
  
  Serial.println("Arduino Servo Controller Ready");
  Serial.println("Waiting for commands...");
}

void loop() {
  // Read serial data
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
  
  // Process the command when a complete string is received
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void processCommand(String command) {
  // Expected format: "SERVO:pin:angle"
  // Example: "SERVO:9:90"
  
  if (command.startsWith("SERVO:")) {
    // Remove "SERVO:" prefix
    String data = command.substring(6);
    
    // Find the colon separator
    int colonIndex = data.indexOf(':');
    
    if (colonIndex > 0) {
      // Extract pin number
      int pin = data.substring(0, colonIndex).toInt();
      
      // Extract angle
      int angle = data.substring(colonIndex + 1).toInt();
      
      // Validate and execute
      if (pin == 9 || pin == 10) {
        if (angle >= 0 && angle <= 180) {
          if (pin == 9) {
            servo9.write(angle);
            Serial.print("Servo 9 moved to ");
            Serial.print(angle);
            Serial.println(" degrees");
          } else if (pin == 10) {
            servo10.write(angle);
            Serial.print("Servo 10 moved to ");
            Serial.print(angle);
            Serial.println(" degrees");
          }
        } else {
          Serial.println("Error: Angle must be between 0 and 180");
        }
      } else {
        Serial.println("Error: Pin must be 9 or 10");
      }
    } else {
      Serial.println("Error: Invalid command format");
    }
  } else {
    Serial.println("Error: Unknown command");
  }
}

