import logging
import time
import threading
import numpy as np
import pyaudio
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Handles real-time audio capturing and processing for detecting emergency sounds
    """
    # Audio parameters
    RATE = 16000  # Sample rate in Hz
    CHUNK = 1024  # Buffer size in samples
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1  # Mono audio
    
    def __init__(self, sound_classifier, volume_controller):
        self.sound_classifier = sound_classifier
        self.volume_controller = volume_controller
        self.is_monitoring = False
        self.detection_threshold = 0.6
        self.last_detection = {}
        self.monitoring_thread = None
        self.audio = pyaudio.PyAudio()
        self.is_simulation_mode = False
        self.active_device_name = "No microphone detected"
        self.available_input_devices = self._get_available_input_devices()
        self.selected_input_device = self._get_default_input_device()
        
    def _get_available_input_devices(self):
        """Get a list of available input devices"""
        input_devices = []
        try:
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    input_devices.append({
                        'index': i,
                        'name': device_info.get('name', f"Device {i}"),
                        'channels': device_info.get('maxInputChannels'),
                        'sample_rate': int(device_info.get('defaultSampleRate'))
                    })
            
            logger.info(f"Found {len(input_devices)} input devices")
            for device in input_devices:
                logger.info(f"Input device: {device['name']} (index: {device['index']})")
                
        except Exception as e:
            logger.error(f"Error getting input devices: {str(e)}")
        
        return input_devices
        
    def _get_default_input_device(self):
        """Get the default input device index"""
        try:
            default_input = self.audio.get_default_input_device_info()
            logger.info(f"Default input device: {default_input.get('name')} (index: {default_input.get('index')})")
            return default_input.get('index')
        except Exception as e:
            logger.warning(f"Could not get default input device: {str(e)}")
            # If no default device, use the first available input device
            if self.available_input_devices:
                logger.info(f"Using first available input device: {self.available_input_devices[0]['name']}")
                return self.available_input_devices[0]['index']
            return None
        
    def update_settings(self, is_monitoring=None, detection_threshold=None):
        """Update the processor settings"""
        if is_monitoring is not None:
            if is_monitoring and not self.is_monitoring:
                self.start_monitoring()
            elif not is_monitoring and self.is_monitoring:
                self.stop_monitoring()
        
        if detection_threshold is not None:
            self.detection_threshold = float(detection_threshold)
            logger.debug(f"Detection threshold updated to {self.detection_threshold}")
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if self.is_monitoring:
            logger.debug("Monitoring already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.debug("Started audio monitoring")
    
    def stop_monitoring(self):
        """Stop the background monitoring"""
        self.is_monitoring = False
        logger.debug("Stopped audio monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread capturing audio"""
        try:
            logger.debug("Audio monitoring started")
            
            # Open the audio stream
            try:
                # Try to get the system's actual audio device first
                # Try more aggressively to find a working microphone
                import subprocess
                
                # On Linux, try to check available audio devices using arecord
                try:
                    logger.info("Checking system audio devices...")
                    audio_devices = subprocess.check_output(["arecord", "-l"], universal_newlines=True)
                    logger.info(f"System audio devices found: {audio_devices}")
                except:
                    logger.warning("Could not check system audio devices with arecord")
                
                # Try to use the selected input device if available
                if self.selected_input_device is not None:
                    logger.info(f"Attempting to open audio stream with device index {self.selected_input_device}")
                    try:
                        device_info = self.audio.get_device_info_by_index(self.selected_input_device)
                        device_name = device_info.get('name', f"Device {self.selected_input_device}")
                        logger.info(f"Opening audio stream with {device_name}")
                        
                        stream = self.audio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            input_device_index=self.selected_input_device,
                            frames_per_buffer=self.CHUNK
                        )
                        self.is_simulation_mode = False
                        self.active_device_name = device_name
                        logger.info(f"Successfully connected to microphone: {device_name}")
                    except Exception as e:
                        logger.warning(f"Could not open selected device: {str(e)}")
                        self.selected_input_device = None
                
                # If no specific device worked, try different methods to find a working microphone
                if self.selected_input_device is None:
                    # First try to enumerate all devices again to make sure we have the latest
                    self.available_input_devices = self._get_available_input_devices()
                    
                    # Try all available input devices until one works
                    for device in self.available_input_devices:
                        try:
                            logger.info(f"Trying input device: {device['name']} (index: {device['index']})")
                            stream = self.audio.open(
                                format=self.FORMAT,
                                channels=min(device['channels'], self.CHANNELS),  # Use the available channels
                                rate=self.RATE,
                                input=True,
                                input_device_index=device['index'],
                                frames_per_buffer=self.CHUNK
                            )
                            self.selected_input_device = device['index']
                            self.active_device_name = device['name']
                            self.is_simulation_mode = False
                            logger.info(f"Successfully connected to microphone: {device['name']}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to open device {device['name']}: {str(e)}")
                
                # If still no device, try without specifying a device
                if self.selected_input_device is None:
                    try:
                        logger.info("Trying to open default audio stream without specifying device")
                        stream = self.audio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK
                        )
                        self.is_simulation_mode = False
                        self.active_device_name = "Default audio device"
                        logger.info("Successfully connected to default microphone")
                    except Exception as e:
                        logger.error(f"Failed to connect to default microphone: {str(e)}")
                        raise e  # Re-raise to be caught by the outer try/except
                
                logger.info("📢 Audio stream opened successfully, listening for emergency sounds...")
            except Exception as e:
                logger.error(f"Failed to open any audio stream: {str(e)}")
                logger.error("❗ Could not connect to any microphone. The system needs microphone access to detect emergency sounds.")
                # Fall back to simulation mode if we can't access the mic
                self._fallback_monitoring_loop()
                return
            
            # Buffer for collecting audio chunks
            buffer_size = 5  # Collect multiple chunks for better detection
            audio_buffer = []
            
            while self.is_monitoring:
                try:
                    # Read audio data from stream
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_buffer.append(data)
                    
                    # Process audio when buffer is full
                    if len(audio_buffer) >= buffer_size:
                        # Convert to numpy array
                        audio_data = b''.join(audio_buffer)
                        numpy_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Process the audio for sound detection
                        self._process_audio(numpy_data)
                        
                        # Reset buffer
                        audio_buffer = []
                        
                    # Short sleep to prevent excessive CPU usage
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error reading audio: {str(e)}")
                    time.sleep(0.5)  # Wait before retrying
            
            # Clean up
            try:
                stream.stop_stream()
                stream.close()
                logger.debug("Audio stream closed")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}")
            self.is_monitoring = False
    
    def _fallback_monitoring_loop(self):
        """Fallback monitoring loop that simulates audio processing"""
        logger.warning("Falling back to simulation mode for audio monitoring")
        self.is_simulation_mode = True
        self.active_device_name = "Simulation Mode (No microphone detected)"
        
        # Demo mode - detect when "DEMO_TRIGGER_SOUND" file exists with enhanced metadata
        trigger_file = os.path.join(os.getcwd(), "DEMO_TRIGGER_SOUND.txt")
        sound_type_file = os.path.join(os.getcwd(), "DEMO_SOUND_TYPE.txt")
        metadata_file = os.path.join(os.getcwd(), "DEMO_SOUND_METADATA.json")
        last_check_time = 0
        
        logger.info("DEMO MODE: Create a file named 'DEMO_TRIGGER_SOUND.txt' to simulate an emergency sound")
        logger.info("You can specify the sound type by creating 'DEMO_SOUND_TYPE.txt' with a sound type like 'siren'")
        
        while self.is_monitoring:
            current_time = time.time()
            
            # Only check for the trigger file every 2 seconds
            if current_time - last_check_time >= 2.0:
                last_check_time = current_time
                
                # Check if the trigger file exists
                if os.path.exists(trigger_file):
                    # Set default values
                    sound_type = "siren"  # Default sound type
                    confidence = 0.95     # Default confidence level
                    duration = 5.0        # Default sound duration in seconds
                    pattern = "wailing"   # Default sound pattern
                    description = "Emergency vehicle siren"  # Default description
                    
                    # Try to read enhanced metadata if available
                    if os.path.exists(metadata_file):
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                
                                # Extract sound properties from metadata
                                if 'sound_type' in metadata and metadata['sound_type'] in self.sound_classifier.EMERGENCY_SOUNDS:
                                    sound_type = metadata['sound_type']
                                
                                if 'confidence' in metadata:
                                    confidence = float(metadata['confidence'])
                                
                                if 'duration' in metadata:
                                    duration = float(metadata['duration'])
                                
                                if 'characteristics' in metadata and 'pattern' in metadata['characteristics']:
                                    pattern = metadata['characteristics']['pattern']
                                
                                if 'characteristics' in metadata and 'description' in metadata['characteristics']:
                                    description = metadata['characteristics']['description']
                                
                                logger.info(f"Using enhanced metadata for sound simulation: {sound_type}, confidence: {confidence:.2f}")
                                
                        except Exception as e:
                            logger.warning(f"Error reading metadata file: {str(e)}")
                    
                    # Fall back to simple sound type file if no metadata or metadata parsing failed
                    elif os.path.exists(sound_type_file):
                        try:
                            with open(sound_type_file, 'r') as f:
                                custom_sound_type = f.read().strip().lower()
                                if custom_sound_type in self.sound_classifier.EMERGENCY_SOUNDS:
                                    sound_type = custom_sound_type
                                    logger.info(f"Using sound type from file: {sound_type}")
                        except Exception as e:
                            logger.warning(f"Error reading sound type file: {str(e)}")
                    
                    # Create simulated sound data that matches the sound type's frequency pattern
                    freq_range = self.sound_classifier.SOUND_FREQUENCIES.get(sound_type, (700, 1600))
                    avg_freq = (freq_range[0] + freq_range[1]) / 2
                    
                    # Generate appropriate simulation based on sound type and pattern
                    # Generate a sine wave at the target frequency
                    t = np.arange(int(self.RATE * min(duration, 1.0))) / self.RATE  # Use a max of 1 second for the waveform
                    audio_data = np.sin(2 * np.pi * avg_freq * t) * 0.7
                    
                    # Apply pattern-specific modulations
                    if sound_type in ['siren', 'ambulance', 'police_car', 'emergency_vehicle'] or pattern == 'wailing':
                        # Add modulation to simulate a siren wail
                        mod_freq = 4  # Hz - how fast the siren wails up and down
                        mod_depth = 200  # Hz - how much the frequency changes
                        fm = avg_freq + mod_depth * np.sin(2 * np.pi * mod_freq * t)
                        audio_data = np.sin(2 * np.pi * fm * t) * 0.7
                    
                    elif sound_type in ['alarm', 'fire_alarm'] or pattern == 'beeping':
                        # Add pulsing to simulate alarm beeps
                        pulse_freq = 5  # Hz - how fast it pulses
                        audio_data *= 0.5 + 0.5 * np.sin(2 * np.pi * pulse_freq * t)**10
                    
                    elif sound_type in ['glass_breaking', 'gun_shot'] or pattern == 'sudden':
                        # Add noise burst with sharp attack
                        noise = np.random.normal(0, 1, len(t))
                        envelope = np.exp(-10 * t)
                        audio_data = noise * envelope
                    
                    elif sound_type in ['scream', 'shouting', 'crying'] or pattern == 'varying':
                        # Simulate human vocal sounds with frequency variations
                        warble_freq = 7  # Hz - how fast the frequency changes
                        warble_depth = 80  # Hz - how much the frequency varies
                        fm = avg_freq + warble_depth * np.sin(2 * np.pi * warble_freq * t) + warble_depth/2 * np.sin(2 * np.pi * warble_freq * 1.5 * t)
                        audio_data = np.sin(2 * np.pi * fm * t) * 0.7
                    
                    # Simulate detecting this specific sound
                    logger.info(f"DEMO: Simulating {sound_type} sound detection")
                    
                    # Record the detection with enhanced metadata
                    timestamp = datetime.now()
                    action_message = f"Volume reduced in response to {description}"
                    
                    self.last_detection = {
                        'sound_type': sound_type,
                        'confidence': confidence,
                        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'simulation': True,
                        'duration': duration,
                        'description': description,
                        'action_taken': action_message
                    }
                    
                    # Take action - reduce volume
                    original_volume = self.volume_controller.get_current_volume()
                    self.volume_controller.reduce_volume()
                    reduced_volume = self.volume_controller.get_current_volume()
                    
                    logger.info(f"DEMO: Volume reduced from {original_volume}% to {reduced_volume}%")
                    
                    # Delete the trigger file so we don't keep triggering
                    try:
                        os.remove(trigger_file)
                    except Exception as e:
                        logger.warning(f"Error removing trigger file: {str(e)}")
                    
                    # Delete metadata file if it exists
                    if os.path.exists(metadata_file):
                        try:
                            os.remove(metadata_file)
                        except Exception as e:
                            logger.warning(f"Error removing metadata file: {str(e)}")
                    
                    # Wait a bit then restore volume
                    wait_time = min(max(duration * 0.8, 2.0), 8.0)  # Wait proportional to sound duration (2-8 seconds)
                    time.sleep(wait_time)
                    
                    # Reset volume
                    self.volume_controller.reset_volume()
                    reset_volume = self.volume_controller.get_current_volume()
                    logger.info(f"DEMO: Volume restored to original level ({reset_volume}%)")
                    logger.info("DEMO: Trigger file removed, ready for next trigger")
                    
                # No trigger file found, just wait
                else:
                    # Send empty audio data to periodically update UI without detections
                    audio_data = np.zeros(1024)
                    self._process_audio(audio_data)
            
            # Sleep a short time
            time.sleep(0.1)
    
    def _process_audio(self, audio_data):
        """Process the audio chunk and detect important sounds"""
        try:
            # Get sound classification results
            results = self.sound_classifier.classify(audio_data)
            
            # Check if any emergency sound is detected with confidence > threshold
            emergency_detected = False
            detected_sound = None
            max_confidence = 0.0
            
            for sound_type, confidence in results:
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_sound = sound_type
                
                # Check if it's an emergency sound with confidence above threshold
                if sound_type in ['siren', 'alarm', 'emergency_vehicle', 'scream', 'glass_breaking'] and confidence > self.detection_threshold:
                    emergency_detected = True
                    self.last_detection = {
                        'sound_type': sound_type,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    logger.info(f"Detected emergency sound: {sound_type} with confidence {confidence:.2f}")
                    break
            
            # Take action if emergency sound detected
            if emergency_detected:
                self.volume_controller.reduce_volume()
                logger.debug("Volume reduced due to emergency sound detection")
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
    
    def get_last_detection(self):
        """Return information about the last detected emergency sound"""
        return self.last_detection
