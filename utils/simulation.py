import numpy as np

class WeatherRewardEnv:
    def __init__(self, y_true, y_pred, storm_threshold=0.5, storm_penalty=5.0):
        """
        Simulates the weather as an environment for RL evaluation.
        y_true, y_pred: Numpy arrays (flattened)
        """
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()
        self.storm_threshold = storm_threshold
        self.storm_penalty = storm_penalty
        
    def calculate_reward(self):
        """
        Reward Function (The Fundamental Goal):
        - Base Reward: Inverse of Absolute Error (higher is better)
        - Storm Penalty: High negative reward if model under-predicts real rain > threshold
        """
        errors = np.abs(self.y_true - self.y_pred)
        rewards = 1.0 / (errors + 0.1) # Small offset to avoid DivisionByZero
        
        # Apply penalty for missing heavy rainfall
        # (Hazard avoidance logic)
        storm_mask = self.y_true > self.storm_threshold
        miss_mask = (self.y_pred < self.storm_threshold) & storm_mask
        
        rewards[miss_mask] -= self.storm_penalty
        
        return rewards, np.sum(rewards)
