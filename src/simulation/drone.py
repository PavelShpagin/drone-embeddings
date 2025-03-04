import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

@dataclass
class DroneState:
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    lat: float
    lng: float
    altitude: float
    time: float

class DroneFlight:
    """
    Simulates a drone's flight with drift and noise
    """
    def __init__(self, start_lat, start_lng, start_altitude, velocity, noise_std=0.1):
        self.lat = start_lat
        self.lng = start_lng
        self.altitude = start_altitude
        self.velocity = np.array(velocity)
        self.noise_std = noise_std
        self.position = np.array([start_lat, start_lng, start_altitude])
        
        # Initialize state history
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
    
    def step(self):
        """Move drone forward one time step with noise"""
        # Add noise to velocity
        noisy_velocity = self.velocity + np.random.normal(0, self.noise_std, 3)
        
        # Update position
        self.position += noisy_velocity
        self.lat, self.lng, self.altitude = self.position
        
        # Store history
        self.position_history.append(self.position.copy())
        self.velocity_history.append(noisy_velocity.copy())
        
        return self
    
    def update_state(self, new_lat, new_lng, new_altitude):
        """
        Update drone's state after correction
        
        Args:
            new_lat: New latitude
            new_lng: New longitude
            new_altitude: New altitude
        """
        # Update position
        self.lat = new_lat
        self.lng = new_lng
        self.altitude = new_altitude
        self.position = np.array([new_lat, new_lng, new_altitude])
        
        # Update velocity based on position change
        if len(self.position_history) > 0:
            position_change = self.position - self.position_history[-1]
            self.velocity = position_change  # New velocity based on correction
        
        # Store updated state
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        
        return self
    
    def get_state(self):
        """Get current state"""
        return self.lat, self.lng, self.altitude, self.velocity 