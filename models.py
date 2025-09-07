from app import db

class Setting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_monitoring = db.Column(db.Boolean, default=True)
    volume_reduction = db.Column(db.Integer, default=50)  # Percentage to reduce volume
    detection_threshold = db.Column(db.Float, default=0.6)  # Confidence threshold for detection
    monitoring_device = db.Column(db.String(50), default="system")  # "system" or "headphones"

class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    sound_type = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    action_taken = db.Column(db.String(200))
