"""
Rule-based maintenance decision logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MaintenanceRules:
    """
    Rule-based system for determining maintenance needs.
    """
    
    def __init__(
        self,
        failure_probability_threshold: float = 0.7,
        rul_threshold_hours: float = 48.0
    ):
        """
        Initialize maintenance rules.
        
        Args:
            failure_probability_threshold: Threshold for failure probability
            rul_threshold_hours: Threshold for RUL in hours
        """
        self.failure_probability_threshold = failure_probability_threshold
        self.rul_threshold_hours = rul_threshold_hours
    
    def evaluate_maintenance_need(
        self,
        failure_probability: float,
        rul_hours: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Evaluate if maintenance is needed based on rules.
        
        Args:
            failure_probability: Predicted failure probability (0-1)
            rul_hours: Predicted RUL in hours (optional)
            
        Returns:
            Dictionary with maintenance decision
        """
        maintenance_needed = False
        urgency = "low"
        reason = []
        
        # Check failure probability
        if failure_probability >= self.failure_probability_threshold:
            maintenance_needed = True
            urgency = "high"
            reason.append(f"High failure probability ({failure_probability:.2%})")
        
        # Check RUL
        if rul_hours is not None and not np.isnan(rul_hours):
            if rul_hours <= self.rul_threshold_hours:
                maintenance_needed = True
                if urgency == "low":
                    urgency = "high"
                reason.append(f"Low RUL ({rul_hours:.1f} hours)")
            elif rul_hours <= self.rul_threshold_hours * 2:
                if not maintenance_needed:
                    maintenance_needed = True
                    urgency = "medium"
                reason.append(f"Moderate RUL ({rul_hours:.1f} hours)")
        
        return {
            'maintenance_needed': maintenance_needed,
            'urgency': urgency,
            'reasons': reason,
            'failure_probability': failure_probability,
            'rul_hours': rul_hours
        }
    
    def evaluate_turbines(
        self,
        turbine_data: pd.DataFrame,
        failure_prob_col: str = 'failure_probability',
        rul_col: str = 'rul_hours'
    ) -> pd.DataFrame:
        """
        Evaluate maintenance needs for multiple turbines.
        
        Args:
            turbine_data: DataFrame with turbine predictions
            failure_prob_col: Column name for failure probability
            rul_col: Column name for RUL
            
        Returns:
            DataFrame with maintenance decisions added
        """
        results = []
        
        for idx, row in turbine_data.iterrows():
            failure_prob = row.get(failure_prob_col, 0.0)
            rul = row.get(rul_col, None)
            
            decision = self.evaluate_maintenance_need(failure_prob, rul)
            decision['turbine_id'] = row.get('turbine_id', idx)
            results.append(decision)
        
        result_df = pd.DataFrame(results)
        
        logger.info(f"Evaluated {len(result_df)} turbines")
        logger.info(f"Maintenance needed: {result_df['maintenance_needed'].sum()}")
        logger.info(f"High urgency: {(result_df['urgency'] == 'high').sum()}")
        
        return result_df

