"""
Squat assessment and tracking module (Sagittal view)
Simplified dataclass-based approach for per-repetition tracking
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SquatRep:
    """Data container for a completed squat repetition"""
    rep_number: int
    min_knee_angle: float
    max_trunk_lean: float
    min_ankle_angle: float
    min_hip_angle: float
    max_heel_lift: float  # As percentage of shin length
    
    def __str__(self):
        return (f"Rep {self.rep_number}: "
                f"Knee={self.min_knee_angle}° | "
                f"Trunk={self.max_trunk_lean}° | "
                f"Ankle={self.min_ankle_angle}° | "
                f"Hip={self.min_hip_angle}° | "
                f"Heel Lift={self.max_heel_lift}%")


class CurrentRep:
    """Tracks angles during the current repetition"""
    
    def __init__(self, rep_number: int):
        self.rep_number = rep_number
        self.angles = {
            'knee': [],
            'trunk_lean': [],
            'ankle': [],
            'hip': [],
            'heel_lift': []  # Normalized as percentage of shin length
        }
    
    def add_angles(self, knee_angle: float, trunk_lean_angle: Optional[float] = None,
                   ankle_angle: Optional[float] = None, hip_angle: Optional[float] = None,
                   heel_lift_percent: Optional[float] = None):
        """Record angles for this frame"""
        self.angles['knee'].append(knee_angle)
        if trunk_lean_angle is not None:
            self.angles['trunk_lean'].append(trunk_lean_angle)
        if ankle_angle is not None:
            self.angles['ankle'].append(ankle_angle)
        if hip_angle is not None:
            self.angles['hip'].append(hip_angle)
        if heel_lift_percent is not None:
            self.angles['heel_lift'].append(heel_lift_percent)
    
    def finalize(self) -> SquatRep:
        """Calculate extremes and return completed rep data"""
        return SquatRep(
            rep_number=self.rep_number,
            min_knee_angle=min(self.angles['knee']) if self.angles['knee'] else 0,
            max_trunk_lean=max(self.angles['trunk_lean'], key=abs) if self.angles['trunk_lean'] else 0,
            min_ankle_angle=min(self.angles['ankle']) if self.angles['ankle'] else 0,
            min_hip_angle=min(self.angles['hip']) if self.angles['hip'] else 0,
            max_heel_lift=max(self.angles['heel_lift']) if self.angles['heel_lift'] else 0
        )


class SquatTracker:
    """
    Simplified squat tracker using dataclass approach
    Tracks squat repetitions and per-rep metrics
    
    Uses velocity-based bottom detection - works with any squat depth
    """
    
    def __init__(self, start_threshold: float = 150, bottom_threshold: float = None):
        """
        Initialize squat tracker
        
        Args:
            start_threshold: Knee angle above this indicates standing position (degrees)
            bottom_threshold: Deprecated - ignored. Uses velocity-based detection instead.
        """
        self.start_threshold = start_threshold
        
        # Current frame values
        self.current_knee_angle = 0
        self.previous_knee_angle = None
        self.current_trunk_lean = 0
        self.current_ankle_angle = 0
        self.current_hip_angle = 0
        self.current_heel_lift = 0  # Normalized percentage
        
        # State tracking
        self.in_squat = False
        self.reached_bottom = False
        self.is_descending = False  # Track if knee is moving down or up
        self.upright_trunk_reference = None
        
        # Rep management
        self.squat_count = 0
        self.current_rep: Optional[CurrentRep] = None
        self.completed_reps: list[SquatRep] = []
    
    def update(self, knee_angle: float, trunk_lean_angle: Optional[float] = None,
               ankle_angle: Optional[float] = None, hip_angle: Optional[float] = None,
               heel_lift_percent: Optional[float] = None) -> bool:
        """
        Update tracker with new frame data
        
        Uses velocity-based bottom detection - detects when knee stops descending and starts ascending
        
        Args:
            knee_angle: Current knee angle in degrees
            trunk_lean_angle: Current trunk lean angle (optional)
            ankle_angle: Current ankle angle (optional)
            hip_angle: Current hip angle (optional)
            heel_lift_percent: Heel lift as percentage of shin length (optional)
        
        Returns:
            True if a squat was just completed, False otherwise
        """
        # Store current frame values
        self.current_knee_angle = knee_angle
        self.current_trunk_lean = trunk_lean_angle or 0
        self.current_ankle_angle = ankle_angle or 0
        self.current_hip_angle = hip_angle or 0
        self.current_heel_lift = heel_lift_percent or 0
        
        squat_completed = False
        
        # Squat state machine with velocity-based detection
        if not self.in_squat:
            # Check if squat is starting (knee drops below threshold)
            if knee_angle < self.start_threshold:
                self.in_squat = True
                self.reached_bottom = False
                self.is_descending = True
                self.current_rep = CurrentRep(self.squat_count + 1)
                
                # Initialize upright trunk reference
                if self.upright_trunk_reference is None and trunk_lean_angle is not None:
                    self.upright_trunk_reference = trunk_lean_angle
        
        # In active squat
        if self.in_squat and self.current_rep is not None:
            # Record angles for this frame
            self.current_rep.add_angles(knee_angle, trunk_lean_angle, ankle_angle, hip_angle, heel_lift_percent)
            
            # Detect velocity (direction of motion)
            if self.previous_knee_angle is not None:
                knee_velocity = knee_angle - self.previous_knee_angle
                
                # Detect transition from descending to ascending (bottom of squat)
                if self.is_descending and knee_velocity > 0:
                    # Stopped descending, now ascending - we've reached the bottom
                    self.reached_bottom = True
                    self.is_descending = False
                elif not self.is_descending and knee_velocity < 0:
                    # Started descending again
                    self.is_descending = True
            
            # Check if squat is complete (back to standing after reaching bottom)
            if self.reached_bottom and knee_angle > self.start_threshold:
                # Finalize and store rep
                completed_rep = self.current_rep.finalize()
                self.completed_reps.append(completed_rep)
                self.squat_count += 1
                squat_completed = True
                
                # Reset state for next rep
                self.in_squat = False
                self.reached_bottom = False
                self.is_descending = False
                self.current_rep = None
        
        # Store previous angle for next velocity calculation
        self.previous_knee_angle = knee_angle
        
        return squat_completed
    
    def get_count(self) -> int:
        """Get total squat count"""
        return self.squat_count
    
    def get_current_angles(self) -> dict:
        """Get current frame angles"""
        return {
            'knee': self.current_knee_angle,
            'trunk_lean': self.current_trunk_lean,
            'ankle': self.current_ankle_angle,
            'hip': self.current_hip_angle,
            'heel_lift': self.current_heel_lift
        }
    
    def get_current_rep_extremes(self) -> Optional[dict]:
        """Get current rep extremes (for in-progress rep)"""
        if self.current_rep is None or not self.current_rep.angles['knee']:
            return None
        
        return {
            'min_knee': min(self.current_rep.angles['knee']),
            'max_trunk_lean': max(self.current_rep.angles['trunk_lean'], key=abs) 
                if self.current_rep.angles['trunk_lean'] else 0,
            'min_ankle': min(self.current_rep.angles['ankle']) 
                if self.current_rep.angles['ankle'] else 0,
            'min_hip': min(self.current_rep.angles['hip']) 
                if self.current_rep.angles['hip'] else 0,
            'max_heel_lift': max(self.current_rep.angles['heel_lift']) 
                if self.current_rep.angles['heel_lift'] else 0
        }
    
    def get_rep(self, rep_number: int) -> Optional[SquatRep]:
        """
        Get data for a specific completed repetition
        
        Args:
            rep_number: Rep number (1-indexed)
        
        Returns:
            SquatRep object or None if not found
        """
        if 1 <= rep_number <= len(self.completed_reps):
            return self.completed_reps[rep_number - 1]
        return None
    
    def get_all_reps(self) -> list[SquatRep]:
        """Get all completed repetitions"""
        return self.completed_reps
    
    def reset(self):
        """Reset tracker for a new session"""
        self.current_knee_angle = 0
        self.previous_knee_angle = None
        self.current_trunk_lean = 0
        self.current_ankle_angle = 0
        self.current_hip_angle = 0
        self.current_heel_lift = 0
        self.in_squat = False
        self.reached_bottom = False
        self.is_descending = False
        self.upright_trunk_reference = None
        self.squat_count = 0
        self.current_rep = None
        self.completed_reps = []
