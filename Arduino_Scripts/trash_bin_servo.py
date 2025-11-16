#!/usr/bin/env python3
"""
Trash Bin Servo Control Script
Controls a servo motor connected to an Arduino based on trash bin type.
"""

import serial
import time
import sys


class TrashBinServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        Initialize the servo controller.
        
        Args:
            port: Serial port where Arduino is connected (e.g., '/dev/ttyUSB0' on Linux, 
                  'COM3' on Windows, '/dev/tty.usbserial-*' on macOS)
            baudrate: Communication speed (default: 9600)
        """
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
    def connect(self):
        """Establish connection with Arduino."""
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"Connected to Arduino on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Close connection with Arduino."""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Disconnected from Arduino")
    
    def _get_servo_pin(self, trash_bin_name):
        """
        Determine servo pin based on trash bin name.
        
        Args:
            trash_bin_name: Name of the trash bin (string)
            
        Returns:
            int: Pin number (9 or 10)
        """
        # Default mapping: You can modify this logic based on your needs
        # For example, if trash bin contains "Green" -> pin 9, else -> pin 10
        if "green" in trash_bin_name.lower():
            return 9
        else:
            return 10
    
    def _send_command(self, pin, angle):
        """
        Send servo command to Arduino.
        
        Format: "SERVO:pin:angle" (e.g., "SERVO:9:90")
        
        Args:
            pin: Servo pin number (9 or 10)
            angle: Angle in degrees (0-180)
        """
        if not self.arduino or not self.arduino.is_open:
            print("Error: Arduino not connected")
            return False
        
        command = f"SERVO:{pin}:{angle}\n"
        try:
            self.arduino.write(command.encode())
            print(f"Sent command: {command.strip()}")
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def move_servo_up(self, trash_bin_name):
        """
        Move servo to 90 degrees (up position).
        
        Args:
            trash_bin_name: Name of the trash bin
        """
        if "compost" not in trash_bin_name.lower():
            print(f"'{trash_bin_name}' does not contain 'compost'. Servo will not move.")
            return False
        
        pin = self._get_servo_pin(trash_bin_name)
        print(f"Moving servo on pin {pin} to 90 degrees (up)")
        return self._send_command(pin, 90)
    
    def move_servo_down(self, trash_bin_name):
        """
        Move servo to 0 degrees (down position).
        
        Args:
            trash_bin_name: Name of the trash bin
        """
        if "compost" not in trash_bin_name.lower():
            print(f"'{trash_bin_name}' does not contain 'compost'. Servo will not move.")
            return False
        
        pin = self._get_servo_pin(trash_bin_name)
        print(f"Moving servo on pin {pin} to 0 degrees (down)")
        return self._send_command(pin, 0)


def main():
    """Example usage of the TrashBinServoController."""
    # Example: "Green composte bin"
    trash_bin = "Green composte bin"
    
    # Initialize controller
    # NOTE: Update the port to match your Arduino's serial port
    # On macOS, it's usually something like '/dev/tty.usbserial-*' or '/dev/tty.usbmodem*'
    # On Windows, it's usually 'COM3', 'COM4', etc.
    # On Linux, it's usually '/dev/ttyUSB0' or '/dev/ttyACM0'
    controller = TrashBinServoController(port='/dev/tty.usbmodem14101')  # Update this!
    
    # Connect to Arduino
    if not controller.connect():
        print("Failed to connect. Please check:")
        print("1. Arduino is connected via USB")
        print("2. Correct serial port is specified")
        print("3. Arduino sketch is uploaded and running")
        return
    
    try:
        # Move servo up
        print("\n--- Moving servo UP ---")
        controller.move_servo_up(trash_bin)
        time.sleep(2)  # Wait for servo to move
        
        # Move servo down
        print("\n--- Moving servo DOWN ---")
        controller.move_servo_down(trash_bin)
        time.sleep(2)  # Wait for servo to move
        
    finally:
        # Disconnect
        controller.disconnect()


if __name__ == "__main__":
    main()

