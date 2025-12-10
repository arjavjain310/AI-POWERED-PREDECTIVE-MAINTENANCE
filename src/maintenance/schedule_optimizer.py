"""
Simple maintenance schedule optimizer.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """
    Simple greedy scheduler for maintenance operations.
    """
    
    def __init__(
        self,
        maintenance_duration_hours: float = 8.0,
        preventive_cost: float = 1000.0,
        corrective_cost: float = 5000.0,
        max_maintenances_per_day: int = 2
    ):
        """
        Initialize scheduler.
        
        Args:
            maintenance_duration_hours: Hours needed per maintenance
            preventive_cost: Cost of preventive maintenance
            corrective_cost: Cost of corrective maintenance
            max_maintenances_per_day: Maximum maintenance operations per day
        """
        self.maintenance_duration_hours = maintenance_duration_hours
        self.preventive_cost = preventive_cost
        self.corrective_cost = corrective_cost
        self.max_maintenances_per_day = max_maintenances_per_day
    
    def create_schedule(
        self,
        turbine_maintenance_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        planning_horizon_days: int = 7
    ) -> pd.DataFrame:
        """
        Create maintenance schedule using greedy heuristic.
        
        Args:
            turbine_maintenance_df: DataFrame with columns:
                - turbine_id
                - maintenance_needed (bool)
                - urgency (str: 'high', 'medium', 'low')
                - rul_hours (float)
                - failure_probability (float)
            start_date: Start date for scheduling (default: today)
            planning_horizon_days: Number of days to plan ahead
            
        Returns:
            DataFrame with schedule
        """
        if start_date is None:
            start_date = datetime.now()
        
        # Filter turbines needing maintenance
        needs_maintenance = turbine_maintenance_df[
            turbine_maintenance_df['maintenance_needed'] == True
        ].copy()
        
        if len(needs_maintenance) == 0:
            logger.info("No turbines need maintenance")
            return pd.DataFrame()
        
        # Sort by urgency and RUL (highest priority first)
        urgency_order = {'high': 3, 'medium': 2, 'low': 1}
        needs_maintenance['urgency_score'] = needs_maintenance['urgency'].map(urgency_order)
        needs_maintenance['priority_score'] = (
            needs_maintenance['urgency_score'] * 1000 +
            (1 / (needs_maintenance['rul_hours'].fillna(1000) + 1)) * 100 +
            needs_maintenance['failure_probability'] * 10
        )
        needs_maintenance = needs_maintenance.sort_values('priority_score', ascending=False)
        
        # Create schedule
        schedule = []
        current_date = start_date
        day_count = 0
        maintenances_today = 0
        
        for idx, row in needs_maintenance.iterrows():
            turbine_id = row['turbine_id']
            urgency = row['urgency']
            rul = row.get('rul_hours', np.nan)
            failure_prob = row.get('failure_probability', 0.0)
            
            # Determine maintenance type
            if urgency == 'high' or (not np.isnan(rul) and rul < 24):
                maintenance_type = 'corrective'
                cost = self.corrective_cost
            else:
                maintenance_type = 'preventive'
                cost = self.preventive_cost
            
            # Move to next day if needed
            if maintenances_today >= self.max_maintenances_per_day:
                day_count += 1
                maintenances_today = 0
                current_date = start_date + timedelta(days=day_count)
            
            # Check if we're within planning horizon
            if day_count >= planning_horizon_days:
                logger.warning(f"Planning horizon exceeded. {len(needs_maintenance) - len(schedule)} turbines not scheduled")
                break
            
            # Add to schedule
            schedule.append({
                'turbine_id': turbine_id,
                'date': current_date.date(),
                'time': current_date.time(),
                'maintenance_type': maintenance_type,
                'duration_hours': self.maintenance_duration_hours,
                'cost': cost,
                'urgency': urgency,
                'rul_hours': rul,
                'failure_probability': failure_prob
            })
            
            maintenances_today += 1
        
        schedule_df = pd.DataFrame(schedule)
        
        logger.info(f"Created schedule for {len(schedule_df)} maintenance operations")
        logger.info(f"Total cost: ${schedule_df['cost'].sum():,.2f}")
        logger.info(f"Preventive: {(schedule_df['maintenance_type'] == 'preventive').sum()}")
        logger.info(f"Corrective: {(schedule_df['maintenance_type'] == 'corrective').sum()}")
        
        return schedule_df
    
    def calculate_savings(
        self,
        schedule_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate cost savings from preventive vs corrective maintenance.
        
        Args:
            schedule_df: Maintenance schedule DataFrame
            
        Returns:
            Dictionary with savings metrics
        """
        if len(schedule_df) == 0:
            return {}
        
        preventive_count = (schedule_df['maintenance_type'] == 'preventive').sum()
        corrective_count = (schedule_df['maintenance_type'] == 'corrective').sum()
        
        total_cost = schedule_df['cost'].sum()
        
        # Estimate what cost would be if all were corrective
        worst_case_cost = len(schedule_df) * self.corrective_cost
        
        savings = worst_case_cost - total_cost
        savings_pct = (savings / worst_case_cost) * 100 if worst_case_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'preventive_count': preventive_count,
            'corrective_count': corrective_count,
            'worst_case_cost': worst_case_cost,
            'savings': savings,
            'savings_percentage': savings_pct
        }

