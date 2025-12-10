"""
Synthetic SCADA data generator for wind turbines in Karnataka, India.

Generates realistic time-series data that simulates normal operation
and component failures with associated RUL labels, specifically tailored
for Karnataka wind farms in districts like Chitradurga, Gadag, and Davangere.

Wind patterns based on Karnataka's monsoon and seasonal variations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Karnataka wind farm districts (5 main districts)
KARNATAKA_DISTRICTS = [
    'Chitradurga',    # Known for strong winds
    'Gadag',          # Moderate to high wind speeds
    'Davangere',      # Good wind resource potential
    'Koppal',         # Emerging wind farm location
    'Vijayapura'      # (Bijapur) Good wind potential
]


class SyntheticSCADAGenerator:
    """
    Generate synthetic wind turbine SCADA data with failure patterns.
    Tailored for Karnataka wind farms (Chitradurga, Gadag, Davangere).
    """
    
    def __init__(
        self,
        num_turbines: int = 10,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        interval_minutes: int = 10,
        random_seed: int = 42,
        region: str = "Karnataka",
        district: Optional[str] = None
    ):
        """
        Initialize the synthetic data generator for Karnataka wind farms.
        
        Args:
            num_turbines: Number of turbines to generate data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval_minutes: Sampling interval in minutes
            random_seed: Random seed for reproducibility
            region: Region name (default: Karnataka)
            district: District name (Chitradurga, Gadag, or Davangere)
        """
        self.num_turbines = num_turbines
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.interval_minutes = interval_minutes
        self.random_seed = random_seed
        self.region = region
        self.district = district or np.random.choice(KARNATAKA_DISTRICTS)
        
        np.random.seed(random_seed)
        
        # Generate timestamps
        self.timestamps = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=f"{interval_minutes}min"
        )
        
        logger.info(f"Initialized generator for {num_turbines} turbines")
        logger.info(f"Region: {self.region}, District: {self.district}")
        logger.info(f"Time range: {self.start_date} to {self.end_date}")
        logger.info(f"Total timestamps: {len(self.timestamps)}")
    
    def _generate_wind_speed(self, base_speed: float = None) -> np.ndarray:
        """
        Generate realistic wind speed with Karnataka-specific patterns.
        
        Karnataka wind patterns:
        - Strong winds during monsoon (June-September): 8-15 m/s
        - Moderate winds during pre/post monsoon: 6-12 m/s
        - Lower winds during winter (Dec-Feb): 4-8 m/s
        - Peak winds in afternoon (2-5 PM)
        """
        n = len(self.timestamps)
        
        # Karnataka-specific base wind speeds by district
        district_bases = {
            'Chitradurga': 9.0,   # Known for strong winds
            'Gadag': 8.5,          # Moderate to high wind speeds
            'Davangere': 8.0,      # Good wind resource potential
            'Koppal': 8.2,         # Emerging wind farm location
            'Vijayapura': 7.8      # (Bijapur) Good wind potential
        }
        base_speed = base_speed or district_bases.get(self.district, 8.5)
        
        # Daily pattern - stronger in afternoon (Karnataka pattern)
        hour_of_day = pd.to_datetime(self.timestamps).hour
        daily_pattern = np.zeros(n)
        for i, hour in enumerate(hour_of_day):
            # Peak wind in afternoon (2-5 PM)
            if 14 <= hour <= 17:
                daily_pattern[i] = 3.0
            elif 10 <= hour <= 13 or 18 <= hour <= 20:
                daily_pattern[i] = 1.5
            else:
                daily_pattern[i] = 0.5
        
        # Seasonal pattern - Karnataka monsoon (June-September)
        month = pd.to_datetime(self.timestamps).month
        seasonal = np.zeros(n)
        for i, m in enumerate(month):
            if 6 <= m <= 9:  # Monsoon months
                seasonal[i] = 4.0  # Strong winds
            elif 3 <= m <= 5 or 10 <= m <= 11:  # Pre/post monsoon
                seasonal[i] = 2.0
            else:  # Winter (Dec-Feb)
                seasonal[i] = -1.0  # Lower winds
        
        # Random noise
        noise = np.random.normal(0, 2.0, n)
        wind_speed = base_speed + daily_pattern + seasonal + noise
        
        # Karnataka wind speed range: 3-20 m/s (realistic for the region)
        return np.clip(wind_speed, 3, 20)
    
    def _generate_power_output(self, wind_speed: np.ndarray) -> np.ndarray:
        """Generate power output based on wind speed (power curve)."""
        # Simplified power curve: P = 0.5 * rho * A * v^3 * Cp
        # Simplified: P ≈ 0.3 * v^3 for v in [3, 12], then capped
        power = np.zeros_like(wind_speed)
        for i, v in enumerate(wind_speed):
            if v < 3:
                power[i] = 0
            elif v < 12:
                power[i] = 0.3 * v ** 3
            elif v < 25:
                power[i] = 0.3 * 12 ** 3  # Rated power
            else:
                power[i] = 0  # Cut-out
        # Add some noise
        power += np.random.normal(0, power * 0.05, len(power))
        return np.clip(power, 0, 2000)  # Max 2MW
    
    def _generate_temperature_features(
        self,
        ambient_temp: np.ndarray,
        power: np.ndarray,
        failure_mode: Optional[str] = None,
        failure_start_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate temperature features with failure patterns.
        Karnataka-specific: Higher ambient temps (25-35°C typical).
        
        Returns:
            Tuple of (gearbox_temp, generator_temp, nacelle_temp)
        """
        n = len(ambient_temp)
        # Karnataka: Higher base temps due to tropical climate
        gearbox_temp = ambient_temp + 18 + power * 0.02
        generator_temp = ambient_temp + 22 + power * 0.025
        nacelle_temp = ambient_temp + 8 + power * 0.01
        
        # Add failure patterns if applicable
        if failure_mode == "gearbox" and failure_start_idx is not None:
            # Gradual temperature increase before failure
            for i in range(failure_start_idx, min(failure_start_idx + 100, n)):
                if i < n:
                    progress = (i - failure_start_idx) / 100
                    gearbox_temp[i] += 10 + 20 * progress
        
        if failure_mode == "generator" and failure_start_idx is not None:
            for i in range(failure_start_idx, min(failure_start_idx + 100, n)):
                if i < n:
                    progress = (i - failure_start_idx) / 100
                    generator_temp[i] += 15 + 25 * progress
        
        # Add noise
        gearbox_temp += np.random.normal(0, 2, n)
        generator_temp += np.random.normal(0, 2.5, n)
        nacelle_temp += np.random.normal(0, 1, n)
        
        return gearbox_temp, generator_temp, nacelle_temp
    
    def _generate_vibration(
        self,
        power: np.ndarray,
        failure_mode: Optional[str] = None,
        failure_start_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate vibration levels with failure patterns.
        
        Returns:
            Tuple of (gearbox_vibration, generator_vibration)
        """
        n = len(power)
        base_vib = 0.5 + power * 0.0005
        gearbox_vib = base_vib + np.random.normal(0, 0.1, n)
        generator_vib = base_vib + np.random.normal(0, 0.1, n)
        
        # Increase vibration before failure
        if failure_mode == "gearbox" and failure_start_idx is not None:
            for i in range(failure_start_idx, min(failure_start_idx + 100, n)):
                if i < n:
                    progress = (i - failure_start_idx) / 100
                    gearbox_vib[i] += 0.5 + 1.5 * progress
        
        if failure_mode == "generator" and failure_start_idx is not None:
            for i in range(failure_start_idx, min(failure_start_idx + 100, n)):
                if i < n:
                    progress = (i - failure_start_idx) / 100
                    generator_vib[i] += 0.5 + 1.5 * progress
        
        return np.clip(gearbox_vib, 0, 10), np.clip(generator_vib, 0, 10)
    
    def _inject_failures(
        self,
        df: pd.DataFrame,
        turbine_id: int
    ) -> pd.DataFrame:
        """
        Inject realistic failure events into the data.
        
        Args:
            df: DataFrame with turbine data
            turbine_id: Turbine identifier
            
        Returns:
            DataFrame with failure labels added
        """
        n = len(df)
        df = df.copy()
        
        # Initialize failure columns
        df['failure_within_horizon'] = 0
        df['failed_component'] = 'none'
        df['time_to_failure_hours'] = np.nan
        
        # Randomly decide if this turbine will have failures
        num_failures = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        
        failure_components = ['gearbox', 'generator', 'bearing']
        failures = []
        
        for _ in range(num_failures):
            # Random failure time (not too early or late)
            failure_idx = np.random.randint(n // 10, n - 200)
            component = np.random.choice(failure_components)
            failures.append((failure_idx, component))
        
        # Sort by failure time
        failures.sort(key=lambda x: x[0])
        
        # Mark failures and compute RUL
        for failure_idx, component in failures:
            # Mark failure point
            df.loc[failure_idx, 'failed_component'] = component
            df.loc[failure_idx, 'status_code'] = 'fault'
            
            # Mark failure within horizon (24 hours = 144 intervals for 10-min data)
            horizon_intervals = 144
            start_idx = max(0, failure_idx - horizon_intervals)
            df.loc[start_idx:failure_idx, 'failure_within_horizon'] = 1
            df.loc[start_idx:failure_idx, 'failed_component'] = component
            
            # Compute RUL (time to failure in hours)
            for i in range(start_idx, failure_idx):
                hours_to_failure = (failure_idx - i) * self.interval_minutes / 60
                df.loc[i, 'time_to_failure_hours'] = hours_to_failure
        
        # Update temperature and vibration based on failures
        for failure_idx, component in failures:
            # Update features leading up to failure
            start_idx = max(0, failure_idx - 100)
            
            if component == 'gearbox':
                for i in range(start_idx, failure_idx):
                    progress = (i - start_idx) / 100
                    df.loc[i, 'gearbox_oil_temperature'] += 10 + 20 * progress
                    df.loc[i, 'vibration_level_gearbox'] += 0.5 + 1.5 * progress
            
            elif component == 'generator':
                for i in range(start_idx, failure_idx):
                    progress = (i - start_idx) / 100
                    df.loc[i, 'generator_temperature'] += 15 + 25 * progress
                    df.loc[i, 'vibration_level_generator'] += 0.5 + 1.5 * progress
            
            elif component == 'bearing':
                for i in range(start_idx, failure_idx):
                    progress = (i - start_idx) / 100
                    df.loc[i, 'vibration_level_gearbox'] += 0.3 + 1.0 * progress
                    df.loc[i, 'vibration_level_generator'] += 0.3 + 1.0 * progress
        
        return df
    
    def generate_turbine_data(self, turbine_id: int) -> pd.DataFrame:
        """
        Generate complete SCADA data for a single turbine.
        
        Args:
            turbine_id: Turbine identifier
            
        Returns:
            DataFrame with all SCADA features and labels
        """
        n = len(self.timestamps)
        
        # Generate base features - Karnataka climate (tropical, 20-35°C typical)
        # Karnataka temperature pattern: warmer year-round
        month = pd.to_datetime(self.timestamps).month
        base_temp = np.zeros(n)
        for i, m in enumerate(month):
            if 3 <= m <= 5:  # Summer (Mar-May): 30-35°C
                base_temp[i] = 32
            elif 6 <= m <= 9:  # Monsoon: 25-30°C
                base_temp[i] = 27
            elif 10 <= m <= 11:  # Post-monsoon: 28-32°C
                base_temp[i] = 30
            else:  # Winter (Dec-Feb): 22-28°C
                base_temp[i] = 25
        
        ambient_temp = base_temp + 3 * np.sin(2 * np.pi * np.arange(n) / (24 * 60 / self.interval_minutes)) + np.random.normal(0, 2, n)
        wind_speed = self._generate_wind_speed()
        wind_direction = np.random.uniform(0, 360, n)
        power_output = self._generate_power_output(wind_speed)
        
        # Rotor and generator speeds (related to wind speed)
        rotor_speed = np.clip(wind_speed * 10 + np.random.normal(0, 2, n), 0, 20)
        generator_speed = rotor_speed * 50 + np.random.normal(0, 10, n)
        
        # Temperatures
        gearbox_temp, generator_temp, nacelle_temp = self._generate_temperature_features(
            ambient_temp, power_output
        )
        
        # Vibration
        gearbox_vib, generator_vib = self._generate_vibration(power_output)
        
        # Control angles
        pitch_angle = np.clip(10 - wind_speed * 0.5 + np.random.normal(0, 1, n), -5, 25)
        yaw_angle = wind_direction + np.random.normal(0, 5, n)
        
        # Status and alarms
        status_code = np.random.choice(['normal', 'normal', 'normal', 'curtailed'], n)
        alarm_code = np.random.choice([0, 1, 2], n, p=[0.85, 0.1, 0.05])
        
        # Create DataFrame with Karnataka region information
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'turbine_id': turbine_id,
            'region': self.region,
            'district': self.district,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'ambient_temperature': ambient_temp,
            'power_output': power_output,
            'rotor_speed': rotor_speed,
            'generator_speed': generator_speed,
            'gearbox_oil_temperature': gearbox_temp,
            'generator_temperature': generator_temp,
            'nacelle_temperature': nacelle_temp,
            'pitch_angle': pitch_angle,
            'yaw_angle': yaw_angle,
            'vibration_level_gearbox': gearbox_vib,
            'vibration_level_generator': generator_vib,
            'alarm_code': alarm_code,
            'status_code': status_code,
        })
        
        # Inject failures
        df = self._inject_failures(df, turbine_id)
        
        return df
    
    def generate_all_data(self, save_path: Optional[str] = None, distribute_districts: bool = True) -> pd.DataFrame:
        """
        Generate data for all turbines across Karnataka districts.
        
        Args:
            save_path: Optional path to save CSV file
            distribute_districts: If True, distribute turbines across districts
            
        Returns:
            Combined DataFrame with all turbines
        """
        logger.info(f"Generating data for {self.num_turbines} turbines in Karnataka...")
        
        all_data = []
        
        if distribute_districts and self.num_turbines >= len(KARNATAKA_DISTRICTS):
            # Distribute turbines across districts
            turbines_per_district = self.num_turbines // len(KARNATAKA_DISTRICTS)
            remainder = self.num_turbines % len(KARNATAKA_DISTRICTS)
            
            turbine_id = 1
            for i, district in enumerate(KARNATAKA_DISTRICTS):
                num_turbines_here = turbines_per_district + (1 if i < remainder else 0)
                if num_turbines_here > 0:  # Only generate if there are turbines for this district
                    self.district = district
                    logger.info(f"Generating {num_turbines_here} turbines for {district} district...")
                    
                    for j in range(num_turbines_here):
                        logger.info(f"  Turbine {turbine_id}/{self.num_turbines} in {district}")
                        df = self.generate_turbine_data(turbine_id)
                        all_data.append(df)
                        turbine_id += 1
        else:
            # All turbines in one district
            for turbine_id in range(1, self.num_turbines + 1):
                logger.info(f"Generating data for turbine {turbine_id}/{self.num_turbines} in {self.district}")
                df = self.generate_turbine_data(turbine_id)
                all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Generated {len(combined_df)} total records")
        logger.info(f"Region: {combined_df['region'].unique()}")
        logger.info(f"Districts: {combined_df['district'].unique()}")
        logger.info(f"Failure rate: {combined_df['failure_within_horizon'].mean():.2%}")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        
        return combined_df


def main():
    """Main function to generate synthetic data for Karnataka wind farms."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Setup logging
    from src.utils.logging_utils import setup_logging
    setup_logging()
    
    # Default paths
    output_path = Path("data/raw/wind_turbine_scada_karnataka.csv")
    
    # Generate data for Karnataka wind farms
    print("=" * 60)
    print("Generating SCADA Data for Karnataka Wind Farms")
    print("Districts: Chitradurga, Gadag, Davangere, Koppal, Vijayapura")
    print("=" * 60)
    
    generator = SyntheticSCADAGenerator(
        num_turbines=10,  # 10 turbines across 5 districts (2 per district)
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval_minutes=10,
        region="Karnataka",
        district=None  # Will be distributed across districts
    )
    
    df = generator.generate_all_data(save_path=str(output_path), distribute_districts=True)
    
    print(f"\n{'='*60}")
    print(f"Data generation complete!")
    print(f"{'='*60}")
    print(f"Total records: {len(df):,}")
    print(f"Turbines: {df['turbine_id'].nunique()}")
    print(f"Region: {df['region'].unique()[0]}")
    print(f"Districts: {', '.join(sorted(df['district'].unique()))}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Failure rate: {df['failure_within_horizon'].mean():.2%}")
    print(f"\nTurbines by district:")
    print(df.groupby('district')['turbine_id'].nunique())
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()

