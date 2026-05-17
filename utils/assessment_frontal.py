"""
Squat assessment and tracking module (Frontal view)
Dataclass-based approach for per-repetition tracking of frontal plane metrics
Tracks knee-heel alignment, trunk centering, and knee symmetry
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrontalSquatRep:
    """Data container for a completed squat repetition (frontal view assessment)"""
    rep_number: int
    left_knee_angle: float
    right_knee_angle: float
    knee_symmetry_score: float  # 0-100%, how symmetric knees are (100 = perfectly symmetric)
    avg_trunk_centering_score: float  # 0-100%, how centered trunk stays during squat
    left_knee_heel_alignment: float  # 0-100%, how well left knee-heel align (100 = perfect)
    right_knee_heel_alignment: float  # 0-100%, how well right knee-heel align (100 = perfect)
    
    def __str__(self):
        return (f"Rep {self.rep_number}: "
                f"L-Knee={self.left_knee_angle}° R-Knee={self.right_knee_angle}° | "
                f"Symmetry={self.knee_symmetry_score:.0f}% | "
                f"Trunk Center={self.avg_trunk_centering_score:.0f}% | "
                f"L-Align={self.left_knee_heel_alignment:.0f}% R-Align={self.right_knee_heel_alignment:.0f}%")


class CurrentFrontalRep:
    """Tracks frontal metrics during the current repetition"""
    
    def __init__(self, rep_number: int):
        self.rep_number = rep_number
        self.metrics = {
            'left_knee_angle': [],
            'right_knee_angle': [],
            'knee_symmetry': [],
            'trunk_centering': [],
            'left_knee_heel_alignment': [],
            'right_knee_heel_alignment': []
        }
    
    def add_metrics(self, left_knee_angle: float, right_knee_angle: float,
                   knee_symmetry: float, trunk_centering: float,
                   left_alignment: float, right_alignment: float):
        """Record frontal metrics for this frame"""
        self.metrics['left_knee_angle'].append(left_knee_angle)
        self.metrics['right_knee_angle'].append(right_knee_angle)
        self.metrics['knee_symmetry'].append(knee_symmetry)
        self.metrics['trunk_centering'].append(trunk_centering)
        self.metrics['left_knee_heel_alignment'].append(left_alignment)
        self.metrics['right_knee_heel_alignment'].append(right_alignment)
    
    def finalize(self) -> FrontalSquatRep:
        """Calculate averages and extremes, return completed rep data"""
        return FrontalSquatRep(
            rep_number=self.rep_number,
            left_knee_angle=min(self.metrics['left_knee_angle']) if self.metrics['left_knee_angle'] else 0,
            right_knee_angle=min(self.metrics['right_knee_angle']) if self.metrics['right_knee_angle'] else 0,
            knee_symmetry_score=sum(self.metrics['knee_symmetry']) / len(self.metrics['knee_symmetry']) 
                if self.metrics['knee_symmetry'] else 0,
            avg_trunk_centering_score=sum(self.metrics['trunk_centering']) / len(self.metrics['trunk_centering']) 
                if self.metrics['trunk_centering'] else 0,
            left_knee_heel_alignment=sum(self.metrics['left_knee_heel_alignment']) / len(self.metrics['left_knee_heel_alignment']) 
                if self.metrics['left_knee_heel_alignment'] else 0,
            right_knee_heel_alignment=sum(self.metrics['right_knee_heel_alignment']) / len(self.metrics['right_knee_heel_alignment']) 
                if self.metrics['right_knee_heel_alignment'] else 0
        )


class FrontalSquatTracker:
    """
    Frontal view squat tracker using dataclass approach
    Tracks squat repetitions and per-rep frontal plane metrics
    
    Monitors knee-heel alignment, trunk centering, and knee symmetry
    Uses velocity-based bottom detection (same as sagittal)
    """
    
    def __init__(self, start_threshold: float = 150):
        """
        Initialize frontal squat tracker
        
        Args:
            start_threshold: Knee angle above this indicates standing position (degrees)
        """
        self.start_threshold = start_threshold
        
        # Current frame values
        self.current_left_knee_angle = 0
        self.current_right_knee_angle = 0
        self.current_knee_symmetry = 0
        self.current_trunk_centering = 0
        self.current_left_alignment = 0
        self.current_right_alignment = 0
        
        # Previous frame for velocity calculation
        self.previous_left_knee_angle = None
        self.previous_right_knee_angle = None
        
        # State tracking
        self.in_squat = False
        self.reached_bottom = False
        self.is_descending = False
        
        # Rep management
        self.squat_count = 0
        self.current_rep: Optional[CurrentFrontalRep] = None
        self.completed_reps: list[FrontalSquatRep] = []
    
    def update(self, left_knee_angle: float, right_knee_angle: float,
               knee_symmetry: float, trunk_centering: float,
               left_alignment: float, right_alignment: float) -> bool:
        """
        Update tracker with new frame data
        
        Uses velocity-based bottom detection on left knee
        
        Args:
            left_knee_angle: Current left knee angle in degrees
            right_knee_angle: Current right knee angle in degrees
            knee_symmetry: Knee symmetry score (0-100%)
            trunk_centering: Trunk centering score (0-100%)
            left_alignment: Left knee-heel alignment score (0-100%)
            right_alignment: Right knee-heel alignment score (0-100%)
        
        Returns:
            True if a squat was just completed, False otherwise
        """
        # Store current frame values
        self.current_left_knee_angle = left_knee_angle
        self.current_right_knee_angle = right_knee_angle
        self.current_knee_symmetry = knee_symmetry
        self.current_trunk_centering = trunk_centering
        self.current_left_alignment = left_alignment
        self.current_right_alignment = right_alignment
        
        squat_completed = False
        
        # Squat state machine with velocity-based detection
        if not self.in_squat:
            # Check if squat is starting (both knees drop below threshold)
            if left_knee_angle < self.start_threshold and right_knee_angle < self.start_threshold:
                self.in_squat = True
                self.reached_bottom = False
                self.is_descending = True
                self.current_rep = CurrentFrontalRep(self.squat_count + 1)
        
        # In active squat
        if self.in_squat and self.current_rep is not None:
            # Record metrics for this frame
            self.current_rep.add_metrics(left_knee_angle, right_knee_angle,
                                        knee_symmetry, trunk_centering,
                                        left_alignment, right_alignment)
            
            # Detect velocity (direction of motion) using left knee
            if self.previous_left_knee_angle is not None:
                left_knee_velocity = left_knee_angle - self.previous_left_knee_angle
                
                # Detect transition from descending to ascending (bottom of squat)
                if self.is_descending and left_knee_velocity > 0:
                    self.reached_bottom = True
                    self.is_descending = False
                elif not self.is_descending and left_knee_velocity < 0:
                    self.is_descending = True
            
            # Check if squat is complete (back to standing after reaching bottom)
            if self.reached_bottom and left_knee_angle > self.start_threshold and right_knee_angle > self.start_threshold:
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
        
        # Store previous angles for next velocity calculation
        self.previous_left_knee_angle = left_knee_angle
        self.previous_right_knee_angle = right_knee_angle
        
        return squat_completed
    
    def get_count(self) -> int:
        """Get total squat count"""
        return self.squat_count
    
    def get_current_metrics(self) -> dict:
        """Get current frame metrics"""
        return {
            'left_knee': self.current_left_knee_angle,
            'right_knee': self.current_right_knee_angle,
            'symmetry': self.current_knee_symmetry,
            'trunk_centering': self.current_trunk_centering,
            'left_alignment': self.current_left_alignment,
            'right_alignment': self.current_right_alignment
        }
    
    def get_current_rep_extremes(self) -> Optional[dict]:
        """Get current rep metrics (for in-progress rep)"""
        if self.current_rep is None or not self.current_rep.metrics['left_knee_angle']:
            return None
        
        return {
            'left_knee': min(self.current_rep.metrics['left_knee_angle']),
            'right_knee': min(self.current_rep.metrics['right_knee_angle']),
            'avg_symmetry': sum(self.current_rep.metrics['knee_symmetry']) / len(self.current_rep.metrics['knee_symmetry']),
            'avg_trunk_centering': sum(self.current_rep.metrics['trunk_centering']) / len(self.current_rep.metrics['trunk_centering']),
            'left_alignment': sum(self.current_rep.metrics['left_knee_heel_alignment']) / len(self.current_rep.metrics['left_knee_heel_alignment']),
            'right_alignment': sum(self.current_rep.metrics['right_knee_heel_alignment']) / len(self.current_rep.metrics['right_knee_heel_alignment'])
        }
    
    def get_rep(self, rep_number: int) -> Optional[FrontalSquatRep]:
        """
        Get data for a specific completed repetition
        
        Args:
            rep_number: Rep number (1-indexed)
        
        Returns:
            FrontalSquatRep object or None if not found
        """
        if 1 <= rep_number <= len(self.completed_reps):
            return self.completed_reps[rep_number - 1]
        return None
    
    def get_all_reps(self) -> list[FrontalSquatRep]:
        """Get all completed repetitions"""
        return self.completed_reps
    
    def reset(self):
        """Reset tracker for a new session"""
        self.current_left_knee_angle = 0
        self.current_right_knee_angle = 0
        self.current_knee_symmetry = 0
        self.current_trunk_centering = 0
        self.current_left_alignment = 0
        self.current_right_alignment = 0
        self.previous_left_knee_angle = None
        self.previous_right_knee_angle = None
        self.in_squat = False
        self.reached_bottom = False
        self.is_descending = False
        self.squat_count = 0
        self.current_rep = None
        self.completed_reps = []
