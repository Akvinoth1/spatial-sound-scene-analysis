import os
import logging
from threading import Thread

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///sound_monitor.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

# Import other components after app initialization to avoid circular imports
from models import Setting

# Initialize database tables
with app.app_context():
    db.create_all()
    
    # Initialize default settings if not exists
    if not Setting.query.first():
        default_settings = Setting(
            is_monitoring=True,
            volume_reduction=50,
            detection_threshold=0.6,
            monitoring_device="system"
        )
        db.session.add(default_settings)
        db.session.commit()
        logger.debug("Initialized default settings")

# Import audio components after DB is set up
from audio_processor import AudioProcessor
from sound_classifier import SoundClassifier
from volume_controller import VolumeController

# Initialize components
sound_classifier = SoundClassifier()
volume_controller = VolumeController()
audio_processor = AudioProcessor(sound_classifier, volume_controller)

# Routes
@app.route('/')
def index():
    settings = Setting.query.first()
    return render_template('index.html', settings=settings)

@app.route('/settings', methods=['POST'])
def update_settings():
    try:
        settings = Setting.query.first()
        
        # Update settings from form
        settings.is_monitoring = 'is_monitoring' in request.form
        settings.volume_reduction = int(request.form.get('volume_reduction', 50))
        settings.detection_threshold = float(request.form.get('detection_threshold', 0.6))
        settings.monitoring_device = request.form.get('monitoring_device', 'system')
        
        db.session.commit()
        
        # Update audio processor settings
        audio_processor.update_settings(
            is_monitoring=settings.is_monitoring,
            detection_threshold=settings.detection_threshold
        )
        
        # Update volume controller settings
        volume_controller.update_settings(
            volume_reduction=settings.volume_reduction,
            device=settings.monitoring_device
        )
        
        flash('Settings updated successfully!', 'success')
    except Exception as e:
        flash(f'Error updating settings: {str(e)}', 'danger')
        logger.error(f"Error updating settings: {str(e)}")
    
    return redirect(url_for('index'))

@app.route('/toggle_monitoring', methods=['POST'])
def toggle_monitoring():
    try:
        settings = Setting.query.first()
        settings.is_monitoring = not settings.is_monitoring
        db.session.commit()
        
        # Update audio processor
        audio_processor.update_settings(is_monitoring=settings.is_monitoring)
        
        status = "started" if settings.is_monitoring else "stopped"
        flash(f'Monitoring {status} successfully!', 'success')
    except Exception as e:
        flash(f'Error toggling monitoring: {str(e)}', 'danger')
        logger.error(f"Error toggling monitoring: {str(e)}")
    
    return redirect(url_for('index'))

@app.route('/api/status')
def get_status():
    settings = Setting.query.first()
    last_detection = audio_processor.get_last_detection()
    
    # Get microphone status for real-time display
    mic_status = {
        'active': not audio_processor.is_simulation_mode,
        'device_name': audio_processor.active_device_name,
        'available_devices': len(audio_processor.available_input_devices),
        'simulation_mode': audio_processor.is_simulation_mode
    }
    
    return jsonify({
        'is_monitoring': settings.is_monitoring,
        'last_detected_sound': last_detection.get('sound_type', 'None'),
        'last_detection_time': last_detection.get('timestamp', 'Never'),
        'current_volume': volume_controller.get_current_volume(),
        'is_volume_reduced': volume_controller.is_volume_reduced,
        'microphone': mic_status
    })

@app.route('/api/trigger_demo_sound', methods=['POST'])
def trigger_demo_sound():
    """API endpoint to trigger a demo sound detection"""
    try:
        # Get sound type from request
        data = request.json
        sound_type = data.get('sound_type', 'siren')
        
        # Validate sound type
        if sound_type not in sound_classifier.EMERGENCY_SOUNDS:
            return jsonify({'error': f'Invalid sound type: {sound_type}'}), 400
            
        # Create the sound type file
        with open("DEMO_SOUND_TYPE.txt", "w") as f:
            f.write(sound_type)
            
        # Create the trigger file
        with open("DEMO_TRIGGER_SOUND.txt", "w") as f:
            f.write("trigger")
            
        logger.info(f"Demo sound triggered: {sound_type}")
        
        return jsonify({'success': True, 'message': f'Triggered {sound_type} sound detection'})
            
    except Exception as e:
        logger.error(f"Error triggering demo sound: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retry_microphone', methods=['POST'])
def retry_microphone():
    """API endpoint to retry connecting to microphone"""
    try:
        # Stop any current monitoring
        audio_processor.stop_monitoring()
        
        # Small delay to ensure cleanup
        import time
        time.sleep(0.5)
        
        # Clear the simulation mode flag to force another attempt at connecting to real hardware
        audio_processor.is_simulation_mode = False
        
        # Reinitialize audio components
        audio_processor.available_input_devices = audio_processor._get_available_input_devices()
        audio_processor.selected_input_device = audio_processor._get_default_input_device()
        
        # Start monitoring again which will try to connect to real hardware
        audio_processor.start_monitoring()
        
        # Get updated mic status
        mic_status = {
            'active': not audio_processor.is_simulation_mode,
            'device_name': audio_processor.active_device_name,
            'available_devices': len(audio_processor.available_input_devices),
            'simulation_mode': audio_processor.is_simulation_mode
        }
        
        logger.info(f"Microphone reconnection attempted. Using real hardware: {not audio_processor.is_simulation_mode}")
        
        return jsonify({
            'success': True, 
            'using_real_microphone': not audio_processor.is_simulation_mode,
            'microphone': mic_status
        })
            
    except Exception as e:
        logger.error(f"Error retrying microphone connection: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start background monitoring in a separate thread
def start_background_monitoring():
    with app.app_context():
        settings = Setting.query.first()
        if settings and settings.is_monitoring:
            audio_processor.start_monitoring()

monitoring_thread = Thread(target=start_background_monitoring)
monitoring_thread.daemon = True
monitoring_thread.start()

logger.debug("Application initialized")
