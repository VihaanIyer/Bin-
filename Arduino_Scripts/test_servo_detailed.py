#!/usr/bin/env python3
"""Detailed servo diagnostic test"""

import serial
import time

PORT = '/dev/tty.usbmodem12301'  # Update this

print("=" * 60)
print("Detailed Servo Diagnostic Test")
print("=" * 60)
print()

try:
    arduino = serial.Serial(PORT, 9600, timeout=2)
    time.sleep(2)
    
    # Clear any initial messages
    while arduino.in_waiting > 0:
        arduino.readline()
    
    print("✅ Arduino connected")
    print()
    print("Testing servo movement with multiple angles...")
    print()
    
    # Test sequence: 0 -> 45 -> 90 -> 135 -> 180 -> 90 -> 0
    angles = [0, 45, 90, 135, 180, 90, 0]
    
    for angle in angles:
        print(f"Moving to {angle} degrees...")
        arduino.write(f"SERVO:{angle}\n".encode())
        time.sleep(0.5)
        
        # Read response
        if arduino.in_waiting > 0:
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            print(f"  Arduino: {response}")
        
        time.sleep(2)  # Wait to see movement
        print()
    
    arduino.close()
    print("=" * 60)
    print("Test complete!")
    print()
    print("Did the servo move?")
    print("  If NO, check the troubleshooting steps below")
    print("=" * 60)
    
except Exception as e:
    print(f"Error: {e}")

