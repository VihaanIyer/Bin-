# Arduino Servo Control Setup

This directory contains scripts for controlling servo motors connected to multiple Arduinos to open/close trash bins automatically.

## Hardware Requirements

- **3 Arduino boards**: 1 Genuino + 2 Uno models
- **3 Servo motors**: Standard SG90 or similar (one per Arduino)
- **USB cables**: One for each Arduino to connect to your computer
- **Power supply**: External 5V power supply recommended for servos (Arduino 5V may not be sufficient for multiple servos)

## Hardware Connections

### For Each Arduino:

1. **Servo Signal Wire** → Digital Pin 9
2. **Servo Power (Red)** → 5V (or external power supply +5V)
3. **Servo Ground (Brown/Black)** → GND
4. **USB Cable** → Connect to computer

### Recommended Power Setup:

- Use an external 5V power supply for servos (shared ground with Arduino)
- Arduino USB provides power for the Arduino board itself
- This prevents voltage drops when multiple servos move simultaneously

## Software Setup

### 1. Install Python Dependencies

```bash
pip install pyserial
```

Or install all requirements:
```bash
pip install -r ../requirements.txt
```

### 2. Upload Arduino Sketch

1. Open Arduino IDE
2. Open `arduino_servo_sketch.ino`
3. **Upload this sketch to EACH Arduino** (Genuino and both Unos)
4. The sketch is identical for all three Arduinos

### 3. Find Your Arduino Ports

#### macOS:
```bash
ls /dev/tty.usbmodem*
ls /dev/tty.usbserial*
```

#### Linux:
```bash
ls /dev/ttyUSB*
ls /dev/ttyACM*
```

#### Windows:
- Open Device Manager → Ports (COM & LPT)
- Look for "Arduino" entries

### 4. Configure Python Script

Edit `trash_bin_servo.py` and update the `arduino_configs` dictionary in the `main()` function:

```python
arduino_configs = {
    'arduino_1': {
        'port': '/dev/tty.usbmodem14101',  # UPDATE THIS
        'servo_pin': 9,
        'description': 'Recycling bin (Blue)'
    },
    'arduino_2': {
        'port': '/dev/tty.usbmodem14102',  # UPDATE THIS
        'servo_pin': 9,
        'description': 'Compost bin (Green)'
    },
    'arduino_3': {
        'port': '/dev/tty.usbmodem14103',  # UPDATE THIS
        'servo_pin': 9,
        'description': 'Landfill bin (Black/Grey)'
    }
}
```

### 5. Test the Setup

Run the test script:
```bash
python trash_bin_servo.py
```

This will:
- Attempt to connect to all 3 Arduinos
- Test each bin by moving the servo up and down
- Show connection status for each Arduino

## Bin-to-Arduino Mapping

The default mapping is:
- **Recycling bin** → `arduino_1`
- **Compost bin** → `arduino_2`
- **Landfill bin** → `arduino_3`

You can change this mapping in the `MultiArduinoServoController` class:

```python
self.bin_to_arduino = {
    'recycling': 'arduino_1',
    'compost': 'arduino_2',
    'landfill': 'arduino_3',
}
```

## Usage in Main Application

To integrate with `main.py`, import and use the controller:

```python
from Arduino_Scripts.trash_bin_servo import MultiArduinoServoController

# Initialize controller
arduino_configs = {
    'arduino_1': {'port': '/dev/tty.usbmodem14101', 'servo_pin': 9},
    'arduino_2': {'port': '/dev/tty.usbmodem14102', 'servo_pin': 9},
    'arduino_3': {'port': '/dev/tty.usbmodem14103', 'servo_pin': 9}
}
servo_controller = MultiArduinoServoController(arduino_configs)
servo_controller.connect_all()

# Open a bin when item is classified
servo_controller.open_bin('recycling', duration=2.0)  # Opens for 2 seconds
```

## Troubleshooting

### Arduino Not Connecting

1. **Check USB connection**: Ensure USB cable is properly connected
2. **Check port**: Verify the port path is correct for your system
3. **Check permissions** (Linux/macOS): You may need to add your user to the `dialout` group:
   ```bash
   sudo usermod -a -G dialout $USER
   # Then log out and back in
   ```
4. **Close other programs**: Make sure Arduino IDE Serial Monitor or other programs aren't using the port

### Servo Not Moving

1. **Check wiring**: Verify servo connections (signal, power, ground)
2. **Check power**: Servo may need external power supply
3. **Check sketch**: Ensure sketch is uploaded to Arduino
4. **Check serial monitor**: Open Arduino IDE Serial Monitor to see if commands are received

### Multiple Arduinos Not Working

1. **Check all USB connections**: Each Arduino needs its own USB cable
2. **Check port paths**: Each Arduino should have a unique port
3. **Test individually**: Test each Arduino separately first
4. **Power supply**: Multiple servos may need external power

## Command Format

The Arduino sketch expects commands in this format:
- `SERVO:0` - Move servo to 0 degrees (closed/down)
- `SERVO:90` - Move servo to 90 degrees (open/up)
- `SERVO:180` - Move servo to 180 degrees (if needed)

## Notes

- All three Arduinos use the same sketch
- Each Arduino controls one servo on pin 9
- The servo pin can be changed in the sketch by modifying `SERVO_PIN`
- Baud rate is 9600 (can be changed in both sketch and Python script if needed)

