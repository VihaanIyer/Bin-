#!/usr/bin/env python3
"""Simple test script to verify Arduino and servo connection"""

import serial
import time

# UPDATE THIS to match one of your Arduino ports
PORT = '/dev/tty.usbmodem12301'  # Change to match your Arduino!

print("=" * 60)
print("Arduino Servo Test")
print("=" * 60)
print(f"Testing port: {PORT}")
print()

try:
    # Connect to Arduino
    print("1. Connecting to Arduino...")
    arduino = serial.Serial(PORT, 9600, timeout=2)
    time.sleep(2)  # Wait for Arduino to initialize
    
    # Read any initial messages from Arduino
    print("   Reading Arduino startup message...")
    time.sleep(0.5)
    while arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"   Arduino says: {line}")
    
    print("   ✅ Connected!")
    print()
    
    # Test 1: Close servo (0 degrees)
    print("2. Testing CLOSE (0 degrees)...")
    arduino.write(b"SERVO:0\n")
    time.sleep(0.5)
    
    # Read response
    if arduino.in_waiting > 0:
        response = arduino.readline().decode('utf-8', errors='ignore').strip()
        print(f"   Arduino response: {response}")
    time.sleep(2)
    print("   ✅ Servo should be CLOSED now")
    print()
    
    # Test 2: Open servo (90 degrees)
    print("3. Testing OPEN (90 degrees)...")
    arduino.write(b"SERVO:90\n")
    time.sleep(0.5)
    
    # Read response
    if arduino.in_waiting > 0:
        response = arduino.readline().decode('utf-8', errors='ignore').strip()
        print(f"   Arduino response: {response}")
    time.sleep(2)
    print("   ✅ Servo should be OPEN now")
    print()
    
    # Test 3: Close again
    print("4. Testing CLOSE again (0 degrees)...")
    arduino.write(b"SERVO:0\n")
    time.sleep(0.5)
    
    # Read response
    if arduino.in_waiting > 0:
        response = arduino.readline().decode('utf-8', errors='ignore').strip()
        print(f"   Arduino response: {response}")
    time.sleep(2)
    print("   ✅ Servo should be CLOSED now")
    print()
    
    # Close connection
    arduino.close()
    print("=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
    print()
    print("What to check:")
    print("  ✓ Did you see Arduino startup messages?")
    print("  ✓ Did you see 'Servo moved to X degrees' responses?")
    print("  ✓ Did the servo physically move?")
    
except serial.SerialException as e:
    print(f"❌ Connection Error: {e}")
    print()
    print("Troubleshooting:")
    print("  1. Check USB cable is connected")
    print("  2. Verify port is correct (check Arduino IDE)")
    print("  3. Close Arduino IDE Serial Monitor (it locks the port)")
    print("  4. Try unplugging and replugging USB cable")
    print("  5. Check permissions: sudo chmod 666 /dev/tty.usbmodem*")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

