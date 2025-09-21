# models.py
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import sqlalchemy as sa # Required for sa.Column types

# Initialize SQLAlchemy here, but it will be bound to the app in app.py
# This allows models to be defined without a full app context initially
db = SQLAlchemy()

# --- Database Model: User ---
class User(db.Model):
    __tablename__ = 'user' # <--- MUST BE DOUBLE UNDERSCORES
    id = db.Column(sa.Integer(), primary_key=True)
    email = db.Column(sa.String(length=200), unique=True, nullable=False)
    password_hash = db.Column(sa.String(length=10000), nullable=False)
    patients = db.relationship('Patient', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self): # <--- MUST BE DOUBLE UNDERSCORES
        return f'<User {self.email}>'

# --- Database Model: Patient ---
class Patient(db.Model):
    __tablename__ = 'patient' # <--- MUST BE DOUBLE UNDERSCORES
    id = db.Column(sa.Integer(), primary_key=True)
    name = db.Column(sa.String(length=100), nullable=False)
    age = db.Column(sa.Integer(), nullable=False)
    sex = db.Column(sa.String(length=10), nullable=False)
    date = db.Column(sa.Date(), nullable=False, default=datetime.utcnow)
    user_id = db.Column(sa.Integer(), db.ForeignKey('user.id'), nullable=False)
    slides = db.relationship('Slide', backref='patient', lazy=True)
    diagnostic_report = db.Column(sa.Text(), nullable=True)
    is_archived = db.Column(sa.Boolean(), default=False, nullable=False)

    def __repr__(self): # <--- MUST BE DOUBLE UNDERSCORES
        return f'<Patient {self.name} (User: {self.user_id})>'

# --- Database Model: Slide ---
class Slide(db.Model):
    __tablename__ = 'slide' # <--- MUST BE DOUBLE UNDERSCORES
    id = db.Column(sa.Integer(), primary_key=True)
    filename = db.Column(sa.String(length=255), nullable=False)
    processed_filename = db.Column(sa.String(length=255), nullable=True)
    biopsy_coords = db.Column(sa.Text(), nullable=True)
    upload_date = db.Column(sa.DateTime(), nullable=False, default=datetime.utcnow)
    patient_id = db.Column(sa.Integer(), db.ForeignKey('patient.id'), nullable=False)

    def __repr__(self): # <--- MUST BE DOUBLE UNDERSCORES
        return f'<Slide {self.filename} (Patient: {self.patient_id})>'
