"""
Visualization functions for wind turbine predictive maintenance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_time_series_around_failure(
    df: pd.DataFrame,
    turbine_id: int,
    failure_time: pd.Timestamp,
    window_hours: int = 48,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot time-series of key features around a failure event.
    
    Args:
        df: DataFrame with SCADA data
        turbine_id: Turbine ID
        failure_time: Timestamp of failure
        window_hours: Hours before/after failure to plot
        features: List of features to plot (default: key features)
        save_path: Path to save plot
    """
    if features is None:
        features = [
            'power_output', 'gearbox_oil_temperature', 'generator_temperature',
            'vibration_level_gearbox', 'vibration_level_generator'
        ]
    
    # Filter data
    turbine_data = df[df['turbine_id'] == turbine_id].copy()
    turbine_data = turbine_data.sort_values('timestamp')
    
    # Get window around failure
    start_time = failure_time - pd.Timedelta(hours=window_hours)
    end_time = failure_time + pd.Timedelta(hours=24)
    
    window_data = turbine_data[
        (turbine_data['timestamp'] >= start_time) &
        (turbine_data['timestamp'] <= end_time)
    ]
    
    if len(window_data) == 0:
        logger.warning(f"No data found for turbine {turbine_id} around failure time")
        return
    
    # Plot
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(features):
        if feature not in window_data.columns:
            continue
        
        axes[i].plot(window_data['timestamp'], window_data[feature], linewidth=2)
        axes[i].axvline(failure_time, color='r', linestyle='--', linewidth=2, label='Failure')
        axes[i].set_ylabel(feature, fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    axes[-1].set_xlabel('Timestamp', fontsize=12)
    plt.suptitle(f'Turbine {turbine_id} - Features Around Failure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time-series plot saved to {save_path}")
    
    plt.close()


def plot_rul_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "RUL Distribution"
):
    """
    Plot distribution of RUL values.
    
    Args:
        df: DataFrame with RUL column
        save_path: Path to save plot
        title: Plot title
    """
    if 'time_to_failure_hours' not in df.columns:
        logger.warning("RUL column not found")
        return
    
    rul = df['time_to_failure_hours'].dropna()
    
    plt.figure(figsize=(10, 6))
    plt.hist(rul, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('RUL (hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"RUL distribution plot saved to {save_path}")
    
    plt.close()


def plot_failure_probability_timeline(
    df: pd.DataFrame,
    turbine_id: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot failure probability over time for a turbine.
    
    Args:
        df: DataFrame with failure probability and timestamp
        turbine_id: Turbine ID
        save_path: Path to save plot
        title: Plot title
    """
    turbine_data = df[df['turbine_id'] == turbine_id].copy()
    turbine_data = turbine_data.sort_values('timestamp')
    
    if 'failure_probability' not in turbine_data.columns:
        logger.warning("Failure probability column not found")
        return
    
    if title is None:
        title = f"Turbine {turbine_id} - Failure Probability Over Time"
    
    plt.figure(figsize=(14, 6))
    plt.plot(turbine_data['timestamp'], turbine_data['failure_probability'], linewidth=2)
    plt.axhline(0.7, color='r', linestyle='--', linewidth=2, label='Threshold (0.7)')
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Failure Probability', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Failure probability plot saved to {save_path}")
    
    plt.close()


def plot_turbine_health_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Turbine Health Comparison"
):
    """
    Plot health comparison across turbines.
    
    Args:
        df: DataFrame with turbine health metrics
        save_path: Path to save plot
        title: Plot title
    """
    if 'health_index' not in df.columns:
        logger.warning("Health index column not found")
        return
    
    health_by_turbine = df.groupby('turbine_id')['health_index'].mean().sort_values()
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(health_by_turbine)), health_by_turbine.values)
    plt.yticks(range(len(health_by_turbine)), health_by_turbine.index)
    plt.xlabel('Average Health Index', fontsize=12)
    plt.ylabel('Turbine ID', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Health comparison plot saved to {save_path}")
    
    plt.close()


def plot_maintenance_schedule(
    schedule_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: str = "Maintenance Schedule"
):
    """
    Plot maintenance schedule as a Gantt chart.
    
    Args:
        schedule_df: DataFrame with maintenance schedule
        save_path: Path to save plot
        title: Plot title
    """
    if len(schedule_df) == 0:
        logger.warning("Empty schedule")
        return
    
    # Convert dates to datetime for plotting
    schedule_df = schedule_df.copy()
    schedule_df['date_dt'] = pd.to_datetime(schedule_df['date'])
    
    # Create color map for urgency
    color_map = {'high': 'red', 'medium': 'orange', 'low': 'green'}
    schedule_df['color'] = schedule_df['urgency'].map(color_map)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(schedule_df) * 0.5)))
    
    y_pos = 0
    for idx, row in schedule_df.iterrows():
        date = row['date_dt']
        duration = row['duration_hours'] / 24  # Convert to days
        color = row['color']
        turbine_id = row['turbine_id']
        
        ax.barh(y_pos, duration, left=date, height=0.8, color=color, alpha=0.7, edgecolor='black')
        ax.text(date, y_pos, f'T{turbine_id}', va='center', ha='left', fontsize=9)
        
        y_pos += 1
    
    ax.set_yticks(range(len(schedule_df)))
    ax.set_yticklabels([f"Turbine {tid}" for tid in schedule_df['turbine_id']])
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Turbine', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='High Urgency'),
        Patch(facecolor='orange', alpha=0.7, label='Medium Urgency'),
        Patch(facecolor='green', alpha=0.7, label='Low Urgency')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Maintenance schedule plot saved to {save_path}")
    
    plt.close()

