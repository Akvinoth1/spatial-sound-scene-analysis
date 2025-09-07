import os
import logging
import numpy as np
import random
from scipy.signal import find_peaks
import librosa

logger = logging.getLogger(__name__)

class SoundClassifier:
    """
    Classifies audio data to detect emergency sounds
    """
    # Define important sound classes
    EMERGENCY_SOUNDS = [
        'siren', 'alarm', 'emergency_vehicle', 'scream', 'glass_breaking',
        'gun_shot', 'car_horn', 'ambulance', 'police_car', 'fire_alarm',
        'announcement', 'shouting', 'crying'
    ]
    
    # Define frequency ranges for different emergency sounds
    SOUND_FREQUENCIES = {
        'siren': (700, 1600),          # Police/ambulance siren frequency range
        'alarm': (2000, 4000),         # High-pitched alarm
        'emergency_vehicle': (700, 1600),  # Similar to siren
        'scream': (1000, 3000),        # Human scream frequencies
        'glass_breaking': (4000, 8000), # High frequency glass breaking
        'car_horn': (400, 800),        # Car horn range
        'fire_alarm': (2800, 3500)     # Standard fire alarm
    }
    
    def __init__(self):
        self.initialized = True
        self.sample_rate = 16000  # Sample rate in Hz
        
        # Energy thresholds for different environments
        self.background_energy_threshold = 0.005  # Threshold to consider audio as not silence
        self.emergency_energy_threshold = 0.02    # Threshold for possible emergency sounds
        
        # Frequency bands for more accurate filtering
        self.freq_bands = {
            'low': (20, 300),
            'mid': (300, 2000),
            'high': (2000, 8000)
        }
        
        # Detection history for temporal smoothing
        self.detection_history = []
        self.history_size = 5  # Keep track of the last 5 detections
        
        logger.info("Enhanced sound classifier initialized")
    
    def extract_features(self, audio_data):
        """Extract audio features for classification"""
        try:
            # Calculate audio energy
            energy = np.mean(np.abs(audio_data))
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract frequency domain features
            # Compute spectrogram
            spec = np.abs(librosa.stft(audio_data))
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(S=spec)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec)[0]
            
            # Compute zero crossing rate
            zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            return {
                'energy': energy,
                'spectral_centroid': np.mean(spectral_centroid),
                'spectral_bandwidth': np.mean(spectral_bandwidth),
                'zero_crossing_rate': np.mean(zero_crossings),
                'spec': spec
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            # Return a simplified feature set
            return {
                'energy': np.mean(np.abs(audio_data)) if audio_data is not None else 0,
                'spectral_centroid': 0,
                'spectral_bandwidth': 0,
                'zero_crossing_rate': 0,
                'spec': np.zeros((1, 1))
            }
    
    def _detect_frequency_pattern(self, audio_data, freq_range):
        """Check if audio contains significant energy in the given frequency range"""
        try:
            # Get the FFT of the audio
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
            
            # Find indices corresponding to the frequency range
            low_idx = np.argmin(np.abs(freqs - freq_range[0]))
            high_idx = np.argmin(np.abs(freqs - freq_range[1]))
            
            # Get the energy in the frequency range
            energy_in_range = np.mean(fft_data[low_idx:high_idx])
            
            # Get the energy in all frequencies
            total_energy = np.mean(fft_data)
            
            # Return the ratio
            return energy_in_range / (total_energy + 1e-10) if total_energy > 0 else 0
            
        except Exception as e:
            logger.error(f"Error in frequency pattern detection: {str(e)}")
            return 0
    
    def _detect_siren_pattern(self, audio_data):
        """Detect siren pattern (oscillating frequencies)"""
        try:
            # Get the spectrogram
            spec = librosa.feature.melspectrogram(y=audio_data, sr=16000)
            
            # Sum across frequency bands
            spec_sum = np.sum(spec, axis=0)
            
            # Find peaks in the summed spectrogram
            peaks, _ = find_peaks(spec_sum, distance=5)
            
            # If we have multiple peaks with regular spacing, it might be a siren
            if len(peaks) >= 3:
                peak_diffs = np.diff(peaks)
                regularity = np.std(peak_diffs) / np.mean(peak_diffs)
                return 1.0 - min(1.0, regularity)  # Higher score for regular spacing
            
            return 0.3  # Default score if not enough peaks
            
        except Exception as e:
            logger.error(f"Error in siren pattern detection: {str(e)}")
            return 0.3
    
    def _analyze_frequency_bands(self, audio_data):
        """Analyze audio in different frequency bands"""
        try:
            # Get the FFT of the audio
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Analyze energy in different frequency bands
            band_energy = {}
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                low_idx = np.argmin(np.abs(freqs - low_freq))
                high_idx = np.argmin(np.abs(freqs - high_freq))
                band_energy[band_name] = np.mean(fft_data[low_idx:high_idx])
            
            # Calculate ratios between bands
            total_energy = sum(band_energy.values()) + 1e-10  # Avoid division by zero
            band_ratios = {band: energy / total_energy for band, energy in band_energy.items()}
            
            return band_energy, band_ratios
            
        except Exception as e:
            logger.error(f"Error in frequency band analysis: {str(e)}")
            return {band: 0 for band in self.freq_bands}, {band: 0.33 for band in self.freq_bands}
    
    def _apply_temporal_smoothing(self, current_detection):
        """Apply temporal smoothing to detections to reduce false positives"""
        if not current_detection:
            self.detection_history.append(None)
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            return []
        
        sound_type, confidence = current_detection[0]  # Get the top detection
        
        # Add current detection to history
        self.detection_history.append((sound_type, confidence))
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # Count occurrences of each sound type in history
        counts = {}
        for item in self.detection_history:
            if item is not None:
                s_type, conf = item
                if s_type not in counts:
                    counts[s_type] = {"count": 0, "total_conf": 0}
                counts[s_type]["count"] += 1
                counts[s_type]["total_conf"] += conf
        
        # Find the most consistent detection
        max_count = 0
        max_type = None
        max_avg_conf = 0
        
        for s_type, data in counts.items():
            if data["count"] > max_count:
                max_count = data["count"]
                max_type = s_type
                max_avg_conf = data["total_conf"] / data["count"]
            elif data["count"] == max_count:
                avg_conf = data["total_conf"] / data["count"]
                if avg_conf > max_avg_conf:
                    max_type = s_type
                    max_avg_conf = avg_conf
        
        # Only return a detection if it appears in at least 2 consecutive frames
        if max_count >= 2:
            # Increase confidence if the detection is very consistent
            consistency_boost = min(max_count / self.history_size, 1.0)
            smoothed_confidence = max_avg_conf * (1 + 0.2 * consistency_boost)
            smoothed_confidence = min(0.95, smoothed_confidence)
            
            return [(max_type, smoothed_confidence)]
        
        return []
    
    def classify(self, audio_data):
        """
        Classify audio data to detect emergency sounds
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            list: List of (sound_type, confidence) tuples
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                return []
            
            # Extract features from audio
            features = self.extract_features(audio_data)
            
            # Initialize results
            results = []
            
            # Check energy level - if too low, likely no significant sound
            if features['energy'] < self.background_energy_threshold:
                return self._apply_temporal_smoothing([])
            
            # Analyze frequency bands
            band_energy, band_ratios = self._analyze_frequency_bands(audio_data)
            
            # Check for each emergency sound
            for sound_type in self.EMERGENCY_SOUNDS:
                confidence = 0.0
                
                # Get frequency range for this sound type
                freq_range = self.SOUND_FREQUENCIES.get(sound_type, (500, 2000))
                
                # Calculate confidence based on frequency pattern
                freq_confidence = self._detect_frequency_pattern(audio_data, freq_range)
                
                # Special handling for different sound types
                if sound_type in ['siren', 'emergency_vehicle', 'ambulance', 'police_car']:
                    # Sirens have oscillating patterns
                    siren_confidence = self._detect_siren_pattern(audio_data)
                    # Sirens typically have more energy in mid-range frequencies
                    mid_energy_factor = min(1.0, band_ratios.get('mid', 0) * 2.5)
                    confidence = max(freq_confidence, siren_confidence) * (0.5 + 0.5 * mid_energy_factor)
                    
                elif sound_type in ['alarm', 'fire_alarm']:
                    # Alarms typically have high energy in high frequencies and are steady
                    high_energy_factor = min(1.0, band_ratios.get('high', 0) * 3.0)
                    # Alarms have high zero-crossing rate
                    if features['zero_crossing_rate'] > 0.2:
                        high_energy_factor *= 1.5
                    confidence = freq_confidence * (0.4 + 0.6 * high_energy_factor)
                    
                elif sound_type in ['scream', 'shouting', 'crying']:
                    # Human vocalizations have energy across mid-high frequencies
                    mid_high_factor = min(1.0, (band_ratios.get('mid', 0) + band_ratios.get('high', 0)) * 2.0)
                    confidence = freq_confidence * (0.3 + 0.7 * mid_high_factor)
                    
                elif sound_type in ['glass_breaking']:
                    # Glass breaking has sharp transients and high frequencies
                    high_energy_factor = min(1.0, band_ratios.get('high', 0) * 4.0)
                    confidence = freq_confidence * (0.2 + 0.8 * high_energy_factor)
                    
                else:
                    confidence = freq_confidence
                
                # Adjust confidence based on energy and spectral features
                if features['energy'] > self.emergency_energy_threshold:  
                    confidence *= 1.2  # Boost confidence for louder sounds
                
                # Scale confidence to 0-1 range
                confidence = min(0.95, confidence)
                
                # Add to results if confidence is reasonable
                if confidence > 0.1:
                    results.append((sound_type, confidence))
            
            # If we couldn't detect any emergency sounds with confidence,
            # don't add random sounds (improves reliability)
            if not results:
                return self._apply_temporal_smoothing([])
            
            # Sort by confidence
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply temporal smoothing to reduce false positives
            return self._apply_temporal_smoothing(results[:3])
            
        except Exception as e:
            logger.error(f"Error during sound classification: {str(e)}")
            return []
