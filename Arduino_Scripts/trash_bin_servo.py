#!/usr/bin/env python3
"""
Trash Bin Servo Control
Controls servo motors on multiple Arduinos based on bin layout JSON.
"""

import serial
import time
import json
import os
from pathlib import Path
from typing import Dict, Optional


class MultiArduinoServoController:
    """Controller for multiple Arduinos, each controlling one servo."""
    
    def __init__(self, arduino_configs: Dict[str, Dict], baudrate=9600):
        """
        Args:
            arduino_configs: {'arduino_1': {'port': '/dev/tty.usbmodem14101'}, ...}
            baudrate: Serial communication speed (default: 9600)
        """
        self.arduino_configs = arduino_configs
        self.baudrate = baudrate
        self.arduinos: Dict[str, serial.Serial] = {}
        
        # Load bin layout and build mapping
        self.bin_layout = self._load_bin_layout()
        self.bin_to_arduino = self._build_mapping()
    
    def _load_bin_layout(self) -> Optional[Dict]:
        """Load bin layout from JSON file."""
        base_dir = Path(__file__).parent.parent
        
        # Try location-specific file
        location = os.getenv('BIN_LOCATION')
        if location:
            path = base_dir / f"bin_layout_{location}.json"
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        
        # Fallback to main metadata
        path = base_dir / "bin_layout_metadata.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _build_mapping(self) -> Dict[str, str]:
        """Map bin types to Arduino names."""
        if not self.bin_layout or 'bins' not in self.bin_layout:
            return {}
        
        mapping = {}
        arduino_names = list(self.arduino_configs.keys())
        
        for i, bin_data in enumerate(self.bin_layout['bins']):
            bin_type = bin_data.get('type', '').lower()
            if bin_type:
                mapping[bin_type] = arduino_names[i % len(arduino_names)]
        
        return mapping
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect to all Arduinos. Returns connection status dict."""
        status = {}
        for name, config in self.arduino_configs.items():
            port = config.get('port')
            if not port:
                status[name] = False
                continue
            try:
                self.arduinos[name] = serial.Serial(port, self.baudrate, timeout=1)
                time.sleep(2)
                status[name] = True
            except:
                status[name] = False
        return status
    
    def disconnect_all(self):
        """Close all Arduino connections."""
        for arduino in self.arduinos.values():
            if arduino and arduino.is_open:
                arduino.close()
        self.arduinos.clear()
    
    def move_servo_up(self, bin_type: str) -> bool:
        """Move servo to 90 degrees (open)."""
        return self._send_command(bin_type, 90)
    
    def move_servo_down(self, bin_type: str) -> bool:
        """Move servo to 0 degrees (closed)."""
        return self._send_command(bin_type, 0)
    
    def open_bin(self, bin_type: str, duration: float = 2.0) -> bool:
        """Open bin, wait, then close."""
        if not self.move_servo_up(bin_type):
            return False
        time.sleep(duration)
        return self.move_servo_down(bin_type)
    
    def _send_command(self, bin_type: str, angle: int) -> bool:
        """Send servo command to Arduino."""
        arduino_name = self.bin_to_arduino.get(bin_type.lower())
        if not arduino_name or arduino_name not in self.arduinos:
            return False
        
        arduino = self.arduinos[arduino_name]
        if not arduino.is_open:
            return False
        
        try:
            arduino.write(f"SERVO:{angle}\n".encode())
            return True
        except:
            return False
