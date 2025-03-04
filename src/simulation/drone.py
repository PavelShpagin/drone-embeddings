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
    def __init__(self, 
                 start_lat: float, 
                 start_lng: float, 
                 start_altitude: float,
                 velocity: np.ndarray,
                 noise_std: float = 0.1,
                 dt: float = 1.0):
        self.states: List[DroneState] = []
        self.dt = dt
        self.noise_std = noise_std
        self.velocity = velocity
        
        # Initialize first state
        self.current_state = DroneState(
            position=np.array([0., 0., start_altitude]),
            velocity=velocity,
            lat=start_lat,
            lng=start_lng,
            altitude=start_altitude,
            time=0.0
        )
        self.states.append(self.current_state)
    
    def step(self) -> DroneState:
        # Add Gaussian noise to velocity
        noisy_velocity = self.velocity + np.random.normal(0, self.noise_std, 3)
        
        # Update position
        new_position = self.current_state.position + noisy_velocity * self.dt
        
        # Update lat/lng based on displacement
        new_lat = self.current_state.lat + noisy_velocity[0] * self.dt / 111111
        new_lng = self.current_state.lng + noisy_velocity[1] * self.dt / (111111 * np.cos(np.radians(new_lat)))
        
        # Create new state
        self.current_state = DroneState(
            position=new_position,
            velocity=noisy_velocity,
            lat=new_lat,
            lng=new_lng,
            altitude=new_position[2],
            time=self.current_state.time + self.dt
        )
        
        self.states.append(self.current_state)
        return self.current_state 