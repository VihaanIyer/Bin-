"""
ElevenLabs Text-to-Speech Handler with Local Fallback
Converts text responses to speech using ElevenLabs or local TTS as fallback
"""

import os
import subprocess
import platform
import threading
from dotenv import load_dotenv

# Load .env from the same directory as this file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
# Also try loading from current directory as fallback
load_dotenv()


class TTSHandler:
    def __init__(self, voice_id=None, language='english'):
        """
        Initialize TTS Handler with ElevenLabs (optional) and local fallback
        
        Args:
            voice_id: Optional specific voice ID to use for ElevenLabs
            language: Language to use ('english' or 'hungarian')
        """
        self.use_elevenlabs = False
        self.client = None
        self.language = language.lower()
        self.voice_id = voice_id
        self.play_func = None
        
        # Interruption support
        self.is_speaking = False
        self.speak_thread = None
        self.audio_process = None  # For subprocess-based TTS
        self._stop_speaking_flag = threading.Event()

        api_key = os.getenv('ELEVENLABS_API_KEY')

        if api_key and api_key != 'your_elevenlabs_api_key_here':
            try:
                from elevenlabs.client import ElevenLabs
                from elevenlabs import play

                self.client = ElevenLabs(api_key=api_key)
                self.play_func = play.play

                # Choose voice
                if not self.voice_id:
                    if self.language == 'hungarian':
                        # Hungarian voice ID
                        self.voice_id = '86V9x9hrQds83qf7zaGn'
                    else:
                        # English default voice (or env override)
                        self.voice_id = os.getenv('ELEVENLABS_VOICE_ID', 'g6xIsTj2HwM6VR4iXFCw')

                # Assume usable; if it's not, we'll fall back on first failure in speak()
                self.use_elevenlabs = True
                print(f"✅ ElevenLabs TTS configured with voice ID: {self.voice_id}")
            except ImportError as e:
                print(f"❌ ElevenLabs package not installed: {e}")
                print("   Install with: pip install elevenlabs")
                self.use_elevenlabs = False
            except Exception as e:
                print(f"❌ ElevenLabs initialization failed, using local TTS: {e}")
                self.use_elevenlabs = False
        else:
            print("⚠️  ElevenLabs API key not found or invalid. Using local TTS fallback.")
        
        # Determine local TTS method based on OS
        self.system = platform.system()
        self.local_tts = None

        if not self.use_elevenlabs:
            if self.system == "Darwin":  # macOS
                print("Using macOS 'say' command for TTS")
            elif self.system == "Linux":
                print("Using 'espeak' for TTS (install with: sudo apt-get install espeak)")
            else:
                print("Using pyttsx3 for TTS")
                try:
                    import pyttsx3
                    self.local_tts = pyttsx3.init()
                except ImportError:
                    print("pyttsx3 not available, TTS may not work")
                    self.local_tts = None
    
    def speak(self, text, save_to_file=None, interruptible=True):
        """
        Convert text to speech and play it (interruptible by default)
        
        Args:
            text: Text to convert to speech
            save_to_file: Optional file path to save audio
            interruptible: If True, can be interrupted by stop_speaking()
        """
        if not text:
            return
        
        # Stop any current speech if interruptible
        if interruptible and self.is_speaking:
            self.stop_speaking()
        
        # Reset stop flag
        self._stop_speaking_flag.clear()
        self.is_speaking = True
        
        # Run TTS in a thread so it can be interrupted
        def _speak_thread():
            try:
                # Try ElevenLabs first
                if self.use_elevenlabs and self.client and self.play_func:
                    try:
                        print(f"🔊 Using ElevenLabs to speak: {text[:60]}...")
                        # Generate audio using ElevenLabs with free tier compatible model
                        audio = self.client.text_to_speech.convert(
                            voice_id=self.voice_id,
                            text=text,
                            model_id="eleven_turbo_v2_5"
                        )

                        # If we need a file, we have to materialize the bytes
                        if save_to_file:
                            chunks = []
                            for chunk in audio:
                                if self._stop_speaking_flag.is_set():
                                    return
                                chunks.append(chunk)
                            audio_bytes = b"".join(chunks)

                            with open(save_to_file, 'wb') as f:
                                f.write(audio_bytes)

                            # Play from bytes (blocking)
                            if not self._stop_speaking_flag.is_set():
                                self.play_func(audio_bytes)
                        else:
                            # Stream directly; playback starts as chunks arrive
                            # Note: play_func is blocking, so we can't easily interrupt it
                            # But we can check the flag before starting
                            if not self._stop_speaking_flag.is_set():
                                self.play_func(audio)

                        if not self._stop_speaking_flag.is_set():
                            print("✅ ElevenLabs audio played successfully")
                        return
                    except Exception as e:
                        print(f"❌ ElevenLabs TTS error, falling back to local: {e}")
                        self.use_elevenlabs = False  # Avoid repeated failures
                
                # Fallback to local TTS
                if self._stop_speaking_flag.is_set():
                    return
                    
                try:
                    if self.system == "Darwin":  # macOS
                        # Use subprocess.Popen so we can kill it
                        self.audio_process = subprocess.Popen(['say', text], 
                                                              stdout=subprocess.PIPE, 
                                                              stderr=subprocess.PIPE)
                        # Wait for process, but check stop flag periodically
                        while self.audio_process.poll() is None:
                            if self._stop_speaking_flag.is_set():
                                self.audio_process.terminate()
                                self.audio_process.wait()
                                return
                            threading.Event().wait(0.1)  # Check every 100ms
                    elif self.system == "Linux":
                        self.audio_process = subprocess.Popen(['espeak', text],
                                                              stdout=subprocess.PIPE,
                                                              stderr=subprocess.PIPE)
                        while self.audio_process.poll() is None:
                            if self._stop_speaking_flag.is_set():
                                self.audio_process.terminate()
                                self.audio_process.wait()
                                return
                            threading.Event().wait(0.1)
                    else:
                        if self.local_tts:
                            if not self._stop_speaking_flag.is_set():
                                self.local_tts.say(text)
                                self.local_tts.runAndWait()
                        else:
                            print(f"TTS: {text}")  # Fallback to print
                except Exception as e:
                    print(f"Local TTS error: {e}")
                    print(f"TTS: {text}")  # Final fallback to print
            finally:
                self.is_speaking = False
                self.audio_process = None
        
        # Start speaking in a thread (non-blocking)
        self.speak_thread = threading.Thread(target=_speak_thread, daemon=True)
        self.speak_thread.start()
        
        # Don't wait - return immediately so caller can listen for interruptions
        # The thread will set is_speaking to False when done
    
    def stop_speaking(self):
        """
        Stop current speech playback immediately
        """
        if not self.is_speaking:
            return
        
        print("🛑 Interrupting speech...")
        self._stop_speaking_flag.set()
        
        # Kill subprocess if it exists
        if self.audio_process and self.audio_process.poll() is None:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=0.5)
            except:
                try:
                    self.audio_process.kill()
                except:
                    pass
        
        # Wait for thread to finish (with timeout)
        if self.speak_thread and self.speak_thread.is_alive():
            self.speak_thread.join(timeout=0.5)
        
        self.is_speaking = False
        print("✅ Speech interrupted")
    
    def get_audio_bytes(self, text):
        """
        Get audio bytes without playing (for async operations)
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes or None
        """
        if not text:
            return None

        if self.use_elevenlabs and self.client:
            try:
                audio = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5"
                )
                # Convert generator to bytes
                return b"".join(audio)
            except Exception as e:
                print(f"Error generating audio bytes: {e}")
                return None
        else:
            # Local TTS doesn't expose raw bytes easily
            return None