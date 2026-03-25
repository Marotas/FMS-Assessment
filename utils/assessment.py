"""Squat assessment and tracking module"""


class SquatTracker:
    """Tracks squat repetitions and metrics"""
    
    def __init__(self, start_threshold=150, bottom_threshold=80):
        """
        Initialize squat tracker
        
        Args:
            start_threshold: Knee angle above this value indicates standing position (degrees)
            bottom_threshold: Knee angle below this value indicates squat bottom (degrees)
        """
        self.squat_count = 0
        self.min_angle = 180  # Track minimum (deepest) squat angle
        self.start_threshold = start_threshold
        self.bottom_threshold = bottom_threshold
        self.in_squat = False  # Track if currently in squat motion
        self.reached_bottom = False  # Track if we've reached the bottom of the squat
    
    def update(self, knee_angle):
        """
        Update tracker with new knee angle reading
        
        Args:
            knee_angle: Current knee angle in degrees
        
        Returns:
            True if a squat was just completed, False otherwise
        """
        # Update min angle (track deepest squat)
        if knee_angle < self.min_angle:
            self.min_angle = knee_angle
        
        squat_completed = False
        
        # State machine for squat detection
        if not self.in_squat:
            # Start of squat: angle must drop below start_threshold (start bending down)
            if knee_angle < self.start_threshold:
                self.in_squat = True
                self.reached_bottom = False
        else:
            # In squat motion
            if knee_angle < self.bottom_threshold:
                # Reached the bottom (deepest part of squat)
                self.reached_bottom = True
            
            if self.reached_bottom and knee_angle > self.start_threshold:
                # Completed squat: went deep and came back up to standing position
                self.squat_count += 1
                squat_completed = True
                self.in_squat = False
                self.reached_bottom = False
        
        return squat_completed
    
    def get_count(self):
        """Get current squat count"""
        return self.squat_count
    
    def get_min_angle(self):
        """Get minimum (deepest) knee angle recorded"""
        return int(self.min_angle) if self.min_angle < 180 else 0
    
    def reset(self):
        """Reset all tracking data"""
        self.squat_count = 0
        self.min_angle = 180
        self.in_squat = False
        self.reached_bottom = False

        self.in_squat = False
        self.reached_bottom = False
