import logging
import time
import threading
import platform
import os
import subprocess

logger = logging.getLogger(__name__)

class VolumeController:
    """
    Controls system or headphone volume with real implementation
    when possible, falling back to simulation when necessary
    """
    def __init__(self):
        self.original_volume = 75  # Starting volume
        self.current_volume = 75   # Current volume
        self.target_reduction = 50  # percentage to reduce
        self.is_volume_reduced = False
        self.reset_timer = None
        self.device = "system"  # "system" or "headphones"
        self.volume_lock = threading.Lock()
        
        # Check if we can access actual volume controls
        self.simulation_mode = True
        try:
            # Try to detect operating system and check volume control access
            self.os_type = platform.system()
            
            if self.os_type == "Linux":
                # Try using ALSA to get volume
                try:
                    subprocess.check_output(["amixer", "-D", "pulse", "sget", "Master"])
                    self.simulation_mode = False
                except:
                    pass
                    
            elif self.os_type == "Darwin":  # macOS
                # Try using osascript to get volume
                try:
                    subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"])
                    self.simulation_mode = False
                except:
                    pass
                    
            elif self.os_type == "Windows":
                # Windows would require more complex checks, default to simulation
                pass
                
        except Exception as e:
            logger.warning(f"Error checking volume control access: {str(e)}")
        
        logger.info(f"Volume controller initialized in {'simulation' if self.simulation_mode else 'real'} mode")
    
    def update_settings(self, volume_reduction=None, device=None):
        """Update volume controller settings"""
        if volume_reduction is not None:
            self.target_reduction = int(volume_reduction)
            logger.debug(f"Volume reduction updated to {self.target_reduction}%")
        
        if device is not None:
            self.device = device
            logger.debug(f"Volume control device updated to {self.device}")
    
    def get_current_volume(self):
        """Get the current system volume"""
        if self.simulation_mode:
            return self.current_volume
            
        try:
            if self.os_type == "Linux":
                cmd = "amixer -D pulse sget Master | grep 'Left:' | awk -F'[][]' '{ print $2 }'"
                volume_str = subprocess.check_output(cmd, shell=True).decode().strip().replace('%', '')
                return int(volume_str)
                
            elif self.os_type == "Darwin":  # macOS
                cmd = "osascript -e 'output volume of (get volume settings)'"
                volume = int(subprocess.check_output(cmd, shell=True).strip())
                return volume
                
            else:
                # Fallback to simulated volume
                return self.current_volume
                
        except Exception as e:
            logger.error(f"Error getting real volume, using simulated value: {str(e)}")
            return self.current_volume
    
    def set_volume(self, volume_level):
        """Set system volume to the specified level (0-100)"""
        try:
            volume_level = max(0, min(100, volume_level))  # Ensure volume is in valid range
            
            if self.simulation_mode:
                self.current_volume = volume_level
                logger.debug(f"Volume set to {volume_level}% (simulated)")
                return True
                
            # Try to set actual system volume
            if self.os_type == "Linux":
                cmd = f"amixer -D pulse sset Master {volume_level}%"
                subprocess.run(cmd, shell=True, check=True)
                
            elif self.os_type == "Darwin":  # macOS
                cmd = f"osascript -e 'set volume output volume {volume_level}'"
                subprocess.run(cmd, shell=True, check=True)
                
            else:
                # Fallback to simulated volume
                self.current_volume = volume_level
                
            logger.debug(f"Volume set to {volume_level}%")
            return True
            
        except Exception as e:
            logger.error(f"Error setting volume: {str(e)}")
            # Fallback to simulation
            self.current_volume = volume_level
            return True
    
    def reduce_volume(self):
        """Reduce volume by the target percentage"""
        with self.volume_lock:
            if self.is_volume_reduced:
                # Volume is already reduced, no need to do it again
                return
            
            # Get current volume and store it
            self.original_volume = self.get_current_volume()
            
            # Calculate reduced volume
            reduced_volume = int(self.original_volume * (1 - self.target_reduction / 100))
            
            # Set the new reduced volume
            if self.set_volume(reduced_volume):
                self.is_volume_reduced = True
                logger.info(f"Volume reduced from {self.original_volume}% to {reduced_volume}%")
                
                # Cancel any existing reset timer
                if self.reset_timer and self.reset_timer.is_alive():
                    self.reset_timer.cancel()
                
                # Set up timer to reset volume after 10 seconds
                self.reset_timer = threading.Timer(10.0, self.reset_volume)
                self.reset_timer.daemon = True
                self.reset_timer.start()
    
    def reset_volume(self):
        """Reset volume to the original level"""
        with self.volume_lock:
            if not self.is_volume_reduced:
                # Volume is not in reduced state
                return
            
            # Restore original volume
            if self.set_volume(self.original_volume):
                self.is_volume_reduced = False
                logger.info(f"Volume restored to original level ({self.original_volume}%)")
