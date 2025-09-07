#!/usr/bin/env python3
"""
Demo script for simulating emergency sounds in the Sound Monitor application.
This allows testing the application by creating temporary files that trigger
sound detection in the simulation mode.
"""

import os
import time
import argparse
import random
import json
from datetime import datetime

# Get the available sound types from the classifier
# This should match the emergency sounds defined in sound_classifier.py
AVAILABLE_SOUNDS = [
    'siren', 'alarm', 'emergency_vehicle', 'scream', 'glass_breaking',
    'gun_shot', 'car_horn', 'ambulance', 'police_car', 'fire_alarm',
    'announcement', 'shouting', 'crying'
]

# Sound characteristics for more realistic simulation
SOUND_PROFILES = {
    'siren': {
        'frequency_range': (700, 1600),
        'duration_range': (5, 15),
        'patterns': ['alternating', 'wailing'],
        'description': 'Police/ambulance siren with alternating high-low pattern'
    },
    'alarm': {
        'frequency_range': (2000, 4000),
        'duration_range': (2, 10),
        'patterns': ['beeping', 'continuous'],
        'description': 'High-pitched alarm tone, typically continuous or rapidly beeping'
    },
    'emergency_vehicle': {
        'frequency_range': (700, 1600),
        'duration_range': (5, 15),
        'patterns': ['alternating', 'wailing'],
        'description': 'Emergency vehicle siren, may include horn sounds'
    },
    'scream': {
        'frequency_range': (1000, 3000),
        'duration_range': (1, 3),
        'patterns': ['sudden', 'varying'],
        'description': 'Human scream with sudden onset and high amplitude'
    },
    'glass_breaking': {
        'frequency_range': (4000, 8000),
        'duration_range': (0.5, 1),
        'patterns': ['sudden', 'crash'],
        'description': 'Sharp, high-frequency sound of glass shattering'
    },
    'gun_shot': {
        'frequency_range': (3000, 5000),
        'duration_range': (0.2, 0.5),
        'patterns': ['sudden', 'sharp'],
        'description': 'Very short, sharp explosive sound'
    },
    'car_horn': {
        'frequency_range': (400, 800),
        'duration_range': (0.5, 2),
        'patterns': ['continuous', 'beeping'],
        'description': 'Mid-range honking sound, often in short bursts'
    },
    'fire_alarm': {
        'frequency_range': (2800, 3500),
        'duration_range': (3, 10),
        'patterns': ['beeping', 'whooping'],
        'description': 'Standard fire alarm with regular pattern'
    }
}

def get_sound_profile(sound_type):
    """Get detailed sound profile for simulation"""
    if sound_type in SOUND_PROFILES:
        profile = SOUND_PROFILES[sound_type].copy()
        # Add some randomness to the simulation
        profile['confidence'] = round(random.uniform(0.70, 0.98), 2)
        profile['duration'] = round(random.uniform(*profile['duration_range']), 1)
        profile['pattern'] = random.choice(profile['patterns'])
        return profile
    else:
        # Default profile for sounds not specifically defined
        return {
            'frequency_range': (500, 2000),
            'duration_range': (1, 5),
            'duration': round(random.uniform(1, 5), 1),
            'patterns': ['varying'],
            'pattern': 'varying',
            'confidence': round(random.uniform(0.70, 0.90), 2),
            'description': 'Generic emergency sound'
        }

def create_simulation_metadata(sound_type):
    """Create detailed simulation metadata for the triggered sound"""
    profile = get_sound_profile(sound_type)
    
    # Create a simulation metadata file with details
    metadata = {
        'sound_type': sound_type,
        'timestamp': datetime.now().isoformat(),
        'simulation': True,
        'confidence': profile['confidence'],
        'duration': profile['duration'],
        'characteristics': {
            'frequency_range': profile['frequency_range'],
            'pattern': profile['pattern'],
            'description': profile.get('description', 'Emergency sound')
        }
    }
    
    # Write metadata to a temporary file
    with open("DEMO_SOUND_METADATA.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def trigger_sound(sound_type=None):
    """Create files to trigger a specific sound detection with enhanced simulation data"""
    # Use a random sound type if none specified
    if not sound_type:
        sound_type = random.choice(AVAILABLE_SOUNDS)
    elif sound_type not in AVAILABLE_SOUNDS:
        print(f"Warning: {sound_type} is not a recognized sound type. Using a random sound.")
        sound_type = random.choice(AVAILABLE_SOUNDS)
    
    # Write the sound type to the monitor file
    with open("DEMO_SOUND_TYPE.txt", "w") as f:
        f.write(sound_type)
    
    # Create enhanced simulation metadata
    metadata = create_simulation_metadata(sound_type)
    
    # Create the trigger file
    with open("DEMO_TRIGGER_SOUND.txt", "w") as f:
        f.write("trigger")
    
    print(f"Simulating {sound_type} detection!")
    print(f"Confidence: {metadata['confidence']:.2f}, Duration: {metadata['duration']}s")
    print(f"Description: {metadata['characteristics']['description']}")
    print("Check the web interface to see the detection response.")
    
    # Wait a bit and then check if the file was removed
    # (it should be removed by the audio processor)
    time.sleep(3)
    if not os.path.exists("DEMO_TRIGGER_SOUND.txt"):
        print("Sound was successfully detected and processed!")
    else:
        print("Warning: Trigger file was not removed. Audio processor might not be running.")
        # Clean up
        try:
            os.remove("DEMO_TRIGGER_SOUND.txt")
        except:
            pass
    
    # Clean up the metadata file
    try:
        os.remove("DEMO_SOUND_METADATA.json")
    except:
        pass

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Trigger emergency sound detection in the Sound Monitor application.')
    parser.add_argument('--sound', '-s', choices=AVAILABLE_SOUNDS, 
                        help='Type of emergency sound to trigger (default: siren)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available sound types')
    
    args = parser.parse_args()
    
    # List available sounds if requested
    if args.list:
        print("Available emergency sound types:")
        for sound in AVAILABLE_SOUNDS:
            print(f"  - {sound}")
        return
    
    # Trigger sound detection
    trigger_sound(args.sound)

if __name__ == "__main__":
    main()