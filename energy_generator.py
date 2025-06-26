
# import numpy as np
# import pandas as pd
# import h5py
# import yaml
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # ====================== CONFIGURATION LOADER ======================
# def load_config(config_path="config.yaml"):
#     """Load configuration from YAML file"""
#     try:
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         print(f"Configuration loaded from {config_path}")
#         return config
#     except FileNotFoundError:
#         print(f"Warning: {config_path} not found. Using default configuration.")
#         return get_default_config()
#     except yaml.YAMLError as e:
#         print(f"Error parsing YAML file: {e}")
#         return get_default_config()

# def get_default_config():
#     """Fallback default configuration if YAML file is not available"""
#     return {
#         'time_resolutions': {'15min': '15T', 'hourly': 'H', 'daily': 'D'},
#         'machines': {
#             "CNC_Mill_1": {"type": "CNC", "base_kWh": 1.2, "operations": ["Milling", "Drilling"]},
#         },
#         'operations': {
#             "Milling": {"kWh_range": [8, 15], "noise": 0.1},
#         },
#         'shifts': {
#             "Morning": {"hours": [6, 14], "label": "Early Shift (6AM-2PM)"},
#         },
#         'public_holidays': ["2024-01-01"],
#         'seasonal_trends': {
#             "Winter": {"base_load_factor": 1.2, "operations_factor": 0.9},
#         },
#         'fault_simulation': {
#             "failure_rate": 0.001,
#             "downtime_hours": {"min": 1, "max": 24},
#             "failure_kWh": 0.0
#         },
#         'data_generation': {
#             "year": 2024,
#             "weekend_activity_factor": {"min": 0.1, "max": 0.5},
#             "daily_operation_hours": {"normal": 16, "reduced": 8}
#         },
#         'visualization': {
#             "default_machine": "CNC_Mill_1",
#             "default_days_to_plot": 7,
#             "colors": {"resolution_15min": "blue", "resolution_hourly": "green", "resolution_daily": "red"}
#         },
#         'file_output': {
#             "hdf5_filename_template": "energy_{year}_advanced_{resolution}.h5"
#         }
#     }

# # ====================== ENERGY DATA GENERATOR CLASS ======================
# class EnergyDataGenerator:
#     def __init__(self, config_path="config.yaml"):
#         """Initialize generator with configuration"""
#         self.config = load_config(config_path)
#         self.machines = self.config['machines']
#         self.operations = self.config['operations']
#         self.shifts = self.config['shifts']
#         self.public_holidays = self.config['public_holidays']
#         self.seasonal_trends = self.config['seasonal_trends']
#         self.fault_params = self.config['fault_simulation']
#         self.data_gen_config = self.config['data_generation']
        
#     def get_shift_info(self, hour):
#         """Determine shift based on hour"""
#         for shift_name, shift_info in self.shifts.items():
#             start, end = shift_info["hours"]
#             if (start <= hour < end) or (shift_name == "Night" and (hour >= 22 or hour < 6)):
#                 return shift_name, shift_info["label"]
#         return "Unknown", "Unknown Shift"

#     def get_season(self, month):
#         """Determine season based on month"""
#         if month in [12, 1, 2]:
#             return "Winter"
#         elif month in [6, 7, 8]:
#             return "Summer"
#         else:
#             return "Spring_Fall"

#     def generate_energy_data(self, resolution='15min'):
#         """Generate energy data at specified resolution"""
#         freq = self.config['time_resolutions'][resolution]
#         year = self.data_gen_config['year']
        
#         if resolution == '15min':
#             dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq=freq)
#         elif resolution == 'hourly':
#             dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq=freq)
#         else:  # daily
#             dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq)
        
#         data = {"timestamp": dates}
        
#         # Add shift and day information
#         shift_names = []
#         shift_labels = []
#         day_names = []
        
#         for ts in dates:
#             # Day name
#             day_names.append(ts.strftime("%A"))
            
#             # Shift information (for daily resolution, use midday hour)
#             if resolution == 'daily':
#                 hour = 12  # Use noon as representative hour for daily data
#             else:
#                 hour = ts.hour
                
#             shift_name, shift_label = self.get_shift_info(hour)
#             shift_names.append(shift_name)
#             shift_labels.append(shift_label)
        
#         data["shift"] = shift_names
#         data["shift_label"] = shift_labels
#         data["day_name"] = day_names
        
#         # Generate machine data
#         for machine, specs in self.machines.items():
#             energy = []
#             for i, ts in enumerate(dates):
#                 # --- Check Context ---
#                 is_weekend = ts.weekday() >= 5
#                 is_holiday = ts.strftime("%Y-%m-%d") in self.public_holidays
                
#                 if resolution == 'daily':
#                     current_hour = 12  # Use noon as representative
#                 else:
#                     current_hour = ts.hour
                
#                 # --- Determine Season ---
#                 season = self.get_season(ts.month)
                
#                 # --- Shift Logic ---
#                 active_shift = None
#                 for shift, shift_info in self.shifts.items():
#                     start, end = shift_info["hours"]
#                     if (start <= current_hour < end) or (shift == "Night" and (current_hour >= 22 or current_hour < 6)):
#                         active_shift = shift
#                         break
                
#                 # --- Energy Calculation ---
#                 if active_shift and not (is_weekend or is_holiday):
#                     # Pick a random operation for the machine
#                     op = np.random.choice(specs["operations"])
#                     op_kWh = np.random.uniform(*self.operations[op]["kWh_range"])
#                     op_kWh *= np.random.normal(1, self.operations[op]["noise"])  # Add noise
                    
#                     # Apply seasonal trend
#                     seasonal_factor = self.seasonal_trends[season]
#                     base_load = specs["base_kWh"] * seasonal_factor["base_load_factor"]
#                     op_kWh *= seasonal_factor["operations_factor"]
                    
#                     total_kWh = base_load + op_kWh
                    
#                     # For 15min resolution, divide hourly consumption by 4
#                     if resolution == '15min':
#                         total_kWh = total_kWh / 4
#                     # For daily resolution, multiply by average daily hours
#                     elif resolution == 'daily':
#                         total_kWh = total_kWh * self.data_gen_config["daily_operation_hours"]["normal"]
                        
#                 else:
#                     # Weekend/Holiday: Reduced activity
#                     activity_factor = np.random.uniform(
#                         self.data_gen_config["weekend_activity_factor"]["min"],
#                         self.data_gen_config["weekend_activity_factor"]["max"]
#                     )
#                     base_consumption = specs["base_kWh"] * activity_factor
                    
#                     if resolution == '15min':
#                         total_kWh = base_consumption / 4
#                     elif resolution == 'daily':
#                         total_kWh = base_consumption * self.data_gen_config["daily_operation_hours"]["reduced"]
#                     else:
#                         total_kWh = base_consumption
                
#                 # --- Simulate Random Failures ---
#                 if np.random.random() < self.fault_params["failure_rate"]:
#                     total_kWh = self.fault_params["failure_kWh"]
                
#                 energy.append(max(0, total_kWh))  # Ensure non-negative values
            
#             data[machine] = np.array(energy)
        
#         return pd.DataFrame(data)

#     def save_to_hdf5(self, df, resolution='15min'):
#         """Save DataFrame to HDF5 file"""
#         filename = self.config['file_output']['hdf5_filename_template'].format(
#             year=self.data_gen_config['year'], 
#             resolution=resolution
#         )
        
#         with h5py.File(filename, "w") as f:
#             # Timestamps (Unix epoch)
#             f.create_dataset("timestamps", data=df["timestamp"].astype(np.int64))
            
#             # Time metadata
#             time_grp = f.create_group("time_info")
#             time_grp.create_dataset("shift", data=np.array(df["shift"], dtype='S20'))
#             time_grp.create_dataset("shift_label", data=np.array(df["shift_label"], dtype='S50'))
#             time_grp.create_dataset("day_name", data=np.array(df["day_name"], dtype='S20'))
            
#             # Machine data (group per machine)
#             machines_grp = f.create_group("machines")
#             for machine in self.machines.keys():
#                 machines_grp.create_dataset(machine, data=df[machine])
            
#             # Metadata (structured groups)
#             meta = f.create_group("metadata")
#             meta.create_dataset("shifts", data=str(self.shifts).encode('utf-8'))
#             meta.create_dataset("holidays", data=np.array(self.public_holidays, dtype='S10'))
#             meta.attrs["resolution"] = resolution
#             meta.attrs["year"] = self.data_gen_config['year']
            
#             # Machine specs (as attributes)
#             for machine, specs in self.machines.items():
#                 machine_grp = machines_grp.create_group(machine + "_specs")
#                 for k, v in specs.items():
#                     if isinstance(v, list):
#                         machine_grp.create_dataset(k, data=np.array(v, dtype='S20'))
#                     else:
#                         machine_grp.attrs[k] = v
            
#             # Operations specs
#             ops_grp = meta.create_group("operations")
#             for op in self.operations.keys():
#                 op_grp = ops_grp.create_group(op)
#                 for param, value in self.operations[op].items():
#                     if isinstance(value, list):  # kWh_range is now a list in YAML
#                         op_grp.create_dataset(param, data=np.array(value))
#                     else:
#                         op_grp.attrs[param] = value
        
#         print(f"Data saved to {filename}")

# # ====================== VISUALIZATION CLASS ======================
# class EnergyVisualizer:
#     def __init__(self, config):
#         """Initialize visualizer with configuration"""
#         self.config = config
#         self.viz_config = config['visualization']
#         self.file_config = config['file_output']
        
#     def plot_energy_trends_all_resolutions(self, df_15min, df_hourly, df_daily, 
#                                          machine=None, days=None):
#         """Create comprehensive plots for all time resolutions"""
#         if machine is None:
#             machine = self.viz_config['default_machine']
#         if days is None:
#             days = self.viz_config['default_days_to_plot']
            
#         fig, axes = plt.subplots(3, 1, figsize=(
#             self.viz_config['figure_size']['width'], 
#             self.viz_config['figure_size']['height']
#         ))
        
#         colors = self.viz_config['colors']
        
#         # 15-minute resolution plot
#         end_idx = days * 24 * 4  # 4 points per hour
#         axes[0].plot(df_15min["timestamp"][:end_idx], df_15min[machine][:end_idx], 
#                      label=f"{machine} - 15min", color=colors['resolution_15min'], alpha=0.7)
#         axes[0].set_title(f"15-Minute Resolution: {machine} Energy Consumption (First {days} Days)")
#         axes[0].set_ylabel("Energy (kWh)")
#         axes[0].grid(True, alpha=0.3)
#         axes[0].legend()
        
#         # Hourly resolution plot
#         end_idx_hourly = days * 24
#         axes[1].plot(df_hourly["timestamp"][:end_idx_hourly], df_hourly[machine][:end_idx_hourly], 
#                      label=f"{machine} - Hourly", color=colors['resolution_hourly'], alpha=0.8)
#         axes[1].set_title(f"Hourly Resolution: {machine} Energy Consumption (First {days} Days)")
#         axes[1].set_ylabel("Energy (kWh)")
#         axes[1].grid(True, alpha=0.3)
#         axes[1].legend()
        
#         # Daily resolution plot
#         end_idx_daily = days
#         axes[2].plot(df_daily["timestamp"][:end_idx_daily], df_daily[machine][:end_idx_daily], 
#                      label=f"{machine} - Daily", color=colors['resolution_daily'], marker='o', alpha=0.8)
#         axes[2].set_title(f"Daily Resolution: {machine} Energy Consumption (First {days} Days)")
#         axes[2].set_ylabel("Energy (kWh)")
#         axes[2].set_xlabel("Time")
#         axes[2].grid(True, alpha=0.3)
#         axes[2].legend()
        
#         plt.tight_layout()
#         filename = self.file_config['plot_filename_template'].format(
#             machine=machine, plot_type=f"all_resolutions_{days}days", resolution="combined"
#         )
#         plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
#                    bbox_inches=self.file_config['bbox_inches'])
#         plt.show()

#     def plot_shift_analysis(self, df, machine=None, resolution='hourly'):
#         """Plot energy consumption by shift"""
#         if machine is None:
#             machine = self.viz_config['default_machine']
            
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Box plot by shift
#         df_sample = df.head(24*7)  # First week
#         sns.boxplot(data=df_sample, x='shift', y=machine, ax=ax1)
#         ax1.set_title(f'{machine} Energy by Shift ({resolution})')
#         ax1.set_ylabel('Energy (kWh)')
        
#         # Average energy by day of week
#         df_sample['day_name'] = pd.Categorical(df_sample['day_name'], 
#                                               categories=['Monday', 'Tuesday', 'Wednesday', 
#                                                         'Thursday', 'Friday', 'Saturday', 'Sunday'],
#                                               ordered=True)
#         day_avg = df_sample.groupby('day_name')[machine].mean()
        
#         ax2.bar(day_avg.index, day_avg.values, color='skyblue', alpha=0.7)
#         ax2.set_title(f'{machine} Average Energy by Day ({resolution})')
#         ax2.set_ylabel('Average Energy (kWh)')
#         ax2.tick_params(axis='x', rotation=45)
        
#         plt.tight_layout()
#         filename = self.file_config['plot_filename_template'].format(
#             machine=machine, plot_type="shift_analysis", resolution=resolution
#         )
#         plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
#                    bbox_inches=self.file_config['bbox_inches'])
#         plt.show()

#     def plot_weekly_heatmap(self, df, machine=None, resolution='hourly'):
#         """Create a weekly heatmap showing energy patterns"""
#         if machine is None:
#             machine = self.viz_config['default_machine']
            
#         # Take first 4 weeks for better visualization
#         df_sample = df.head(24*7*4 if resolution == 'hourly' else 
#                            (24*4*7*4 if resolution == '15min' else 28))
        
#         if resolution == 'hourly':
#             df_sample['hour'] = df_sample['timestamp'].dt.hour
#             pivot_data = df_sample.groupby(['day_name', 'hour'])[machine].mean().unstack()
#         elif resolution == '15min':
#             df_sample['hour_15min'] = df_sample['timestamp'].dt.hour + df_sample['timestamp'].dt.minute/60
#             # Group into hour bins for visualization
#             df_sample['hour'] = df_sample['timestamp'].dt.hour
#             pivot_data = df_sample.groupby(['day_name', 'hour'])[machine].mean().unstack()
#         else:  # daily
#             pivot_data = df_sample.groupby('day_name')[machine].mean().to_frame().T
        
#         # Reorder days
#         day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#         pivot_data = pivot_data.reindex(day_order)
        
#         plt.figure(figsize=(12, 6))
#         sns.heatmap(pivot_data, annot=False, cmap=self.viz_config['heatmap_colormap'], 
#                    cbar_kws={'label': 'Energy (kWh)'})
#         plt.title(f'{machine} Energy Consumption Heatmap - {resolution.title()} Resolution')
#         plt.ylabel('Day of Week')
#         if resolution != 'daily':
#             plt.xlabel('Hour of Day')
#         plt.tight_layout()
#         filename = self.file_config['plot_filename_template'].format(
#             machine=machine, plot_type="heatmap", resolution=resolution
#         )
#         plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
#                    bbox_inches=self.file_config['bbox_inches'])
#         plt.show()

# # ====================== MAIN EXECUTION ======================
# def main():
#     """Main execution function"""
#     print("=== Energy Data Generator with YAML Configuration ===")
    
#     # Initialize generator and visualizer
#     generator = EnergyDataGenerator("config.yaml")
#     visualizer = EnergyVisualizer(generator.config)
    
#     print("Generating advanced energy datasets for all resolutions...")
    
#     # Generate data for all resolutions
#     df_15min = generator.generate_energy_data('15min')
#     df_hourly = generator.generate_energy_data('hourly')
#     df_daily = generator.generate_energy_data('daily')
    
#     print(f"15-minute data shape: {df_15min.shape}")
#     print(f"Hourly data shape: {df_hourly.shape}")
#     print(f"Daily data shape: {df_daily.shape}")
    
#     # Save to HDF5
#     generator.save_to_hdf5(df_15min, '15min')
#     generator.save_to_hdf5(df_hourly, 'hourly')
#     generator.save_to_hdf5(df_daily, 'daily')
#     print("All datasets saved to HDF5 files.")
    
#     # Display sample data
#     print("\nSample 15-minute data:")
#     default_machine = generator.config['visualization']['default_machine']
#     print(df_15min[['timestamp', 'shift', 'day_name', default_machine]].head(10))
    
#     # Create visualizations
#     print("\nGenerating plots...")
    
#     # 1. Multi-resolution comparison
#     visualizer.plot_energy_trends_all_resolutions(df_15min, df_hourly, df_daily)
    
#     # 2. Shift analysis for each resolution
#     for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
#         visualizer.plot_shift_analysis(df, resolution=resolution)
        
#     # 3. Weekly heatmaps
#     for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
#         visualizer.plot_weekly_heatmap(df, resolution=resolution)
    
#     print("All visualizations complete!")
    
#     # Summary statistics
#     print("\n=== SUMMARY STATISTICS ===")
#     for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
#         print(f"\n{resolution.upper()} RESOLUTION:")
#         print(f"Data points: {len(df)}")
#         print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
#         print(f"{default_machine} average consumption: {df[default_machine].mean():.2f} kWh")
#         print(f"Shift distribution:")
#         print(df['shift'].value_counts())

# if __name__ == "__main__":
#     main()
import numpy as np
import pandas as pd
import h5py
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

# ====================== CONFIGURATION LOADER ======================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default configuration.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return get_default_config()

def get_default_config():
    """Fallback default configuration if YAML file is not available"""
    return {
        'time_resolutions': {'15min': '15T', 'hourly': 'H', 'daily': 'D'},
        'machines': {
            "CNC_Mill_1": {"type": "CNC", "base_kWh": 1.2, "operations": ["Milling", "Drilling"]},
        },
        'operations': {
            "Milling": {"kWh_range": [8, 15], "noise": 0.1},
        },
        'shifts': {
            "Morning": {"hours": [6, 14], "label": "Early Shift (6AM-2PM)"},
        },
        'public_holidays': ["2024-01-01"],
        'seasonal_trends': {
            "Winter": {"base_load_factor": 1.2, "operations_factor": 0.9},
        },
        'fault_simulation': {
            "failure_rate": 0.001,
            "downtime_hours": {"min": 1, "max": 24},
            "failure_kWh": 0.0
        },
        'data_generation': {
            "year": 2024,
            "weekend_activity_factor": {"min": 0.1, "max": 0.5},
            "daily_operation_hours": {"normal": 16, "reduced": 8}
        },
        'visualization': {
            "default_machine": "CNC_Mill_1",
            "default_days_to_plot": 7,
            "colors": {"resolution_15min": "blue", "resolution_hourly": "green", "resolution_daily": "red"}
        },
        'file_output': {
            "hdf5_filename_template": "energy_{year}_advanced_{resolution}.h5",
            "metadata_filename_template": "energy_{year}_metadata_{resolution}.json",
            "export_metadata": True
        },
        'validation': {
            "enabled": True,
            "check_negative_values": True,
            "energy_range_validation": {"min_kwh": 0.0, "max_kwh": 1000.0}
        }
    }

# ====================== ENERGY DATA GENERATOR CLASS ======================
class EnergyDataGenerator:
    def __init__(self, config_path="config.yaml"):
        """Initialize generator with configuration"""
        self.config = load_config(config_path)
        self.machines = self.config['machines']
        self.operations = self.config['operations']
        self.shifts = self.config['shifts']
        self.public_holidays = self.config['public_holidays']
        self.seasonal_trends = self.config['seasonal_trends']
        self.fault_params = self.config['fault_simulation']
        self.data_gen_config = self.config['data_generation']
        self.validation_config = self.config.get('validation', {})
        
    def validate_data(self, df, resolution):
        """Validate generated data for consistency and quality"""
        if not self.validation_config.get('enabled', True):
            return True, []
        
        issues = []
        
        # Check for negative values
        if self.validation_config.get('check_negative_values', True):
            for machine in self.machines.keys():
                neg_count = (df[machine] < 0).sum()
                if neg_count > 0:
                    issues.append(f"{machine}: {neg_count} negative values found")
        
        # Check energy range validation
        energy_range = self.validation_config.get('energy_range_validation', {})
        min_kwh = energy_range.get('min_kwh', 0.0)
        max_kwh = energy_range.get('max_kwh', 1000.0)
        
        for machine in self.machines.keys():
            out_of_range = ((df[machine] < min_kwh) | (df[machine] > max_kwh)).sum()
            if out_of_range > 0:
                issues.append(f"{machine}: {out_of_range} values outside range [{min_kwh}, {max_kwh}] kWh")
        
        # Check for missing timestamps
        if self.validation_config.get('check_missing_timestamps', True):
            expected_freq = self.config['time_resolutions'][resolution]
            expected_range = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq=expected_freq)
            if len(df) != len(expected_range):
                issues.append(f"Missing timestamps: expected {len(expected_range)}, got {len(df)}")
        
        # Check shift consistency
        if self.validation_config.get('check_shift_consistency', True):
            for i, row in df.head(100).iterrows():  # Sample check
                expected_shift, _ = self.get_shift_info(row['timestamp'].hour if resolution != 'daily' else 12)
                if row['shift'] != expected_shift and resolution != 'daily':
                    issues.append(f"Shift inconsistency at {row['timestamp']}: expected {expected_shift}, got {row['shift']}")
                    break
        
        # Statistical outlier detection
        if self.validation_config.get('statistical_checks', {}).get('outlier_detection', True):
            threshold = self.validation_config.get('statistical_checks', {}).get('outlier_threshold', 3.0)
            for machine in self.machines.keys():
                z_scores = np.abs(stats.zscore(df[machine]))
                outliers = (z_scores > threshold).sum()
                if outliers > len(df) * 0.05:  # More than 5% outliers is concerning
                    issues.append(f"{machine}: {outliers} statistical outliers (>{threshold} std devs)")
        
        return len(issues) == 0, issues
    
    def generate_metadata(self, df, resolution):
        """Generate comprehensive metadata for the dataset"""
        metadata = {
            "dataset_info": {
                "resolution": resolution,
                "year": self.data_gen_config['year'],
                "generation_timestamp": datetime.now().isoformat(),
                "total_records": len(df),
                "date_range": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat()
                }
            },
            "configuration": {
                "machines": self.machines,
                "operations": self.operations,
                "shifts": self.shifts,
                "seasonal_trends": self.seasonal_trends,
                "fault_simulation": self.fault_params,
                "public_holidays": self.public_holidays
            },
            "data_statistics": {},
            "data_quality": {
                "validation_enabled": self.validation_config.get('enabled', False),
                "validation_timestamp": datetime.now().isoformat()
            }
        }
        
        # Calculate statistics for each machine
        for machine in self.machines.keys():
            machine_data = df[machine]
            metadata["data_statistics"][machine] = {
                "mean": float(machine_data.mean()),
                "std": float(machine_data.std()),
                "min": float(machine_data.min()),
                "max": float(machine_data.max()),
                "median": float(machine_data.median()),
                "total_consumption": float(machine_data.sum()),
                "zero_consumption_periods": int((machine_data == 0).sum()),
                "percentiles": {
                    "25th": float(machine_data.quantile(0.25)),
                    "75th": float(machine_data.quantile(0.75)),
                    "95th": float(machine_data.quantile(0.95))
                }
            }
        
        # Shift distribution
        metadata["data_statistics"]["shift_distribution"] = df['shift'].value_counts().to_dict()
        metadata["data_statistics"]["day_distribution"] = df['day_name'].value_counts().to_dict()
        
        # Seasonal statistics
        df_temp = df.copy()
        df_temp['month'] = df_temp['timestamp'].dt.month
        df_temp['season'] = df_temp['month'].apply(lambda x: self.get_season(x))
        
        for machine in self.machines.keys():
            seasonal_stats = df_temp.groupby('season')[machine].agg(['mean', 'std']).to_dict()
            metadata["data_statistics"][machine]["seasonal_patterns"] = seasonal_stats
        
        return metadata
    
    def save_metadata(self, metadata, resolution):
        """Save metadata to file"""
        if not self.config['file_output'].get('export_metadata', True):
            return
        
        filename = self.config['file_output']['metadata_filename_template'].format(
            year=self.data_gen_config['year'], 
            resolution=resolution
        )
        
        metadata_format = self.config['file_output'].get('metadata_format', 'json')
        
        if metadata_format in ['json', 'both']:
            json_filename = filename
            with open(json_filename, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Metadata saved to {json_filename}")
        
        if metadata_format in ['yaml', 'both']:
            yaml_filename = filename.replace('.json', '.yaml')
            with open(yaml_filename, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False, default_style=None)
            print(f"Metadata saved to {yaml_filename}")
        
    def get_shift_info(self, hour):
        """Determine shift based on hour"""
        for shift_name, shift_info in self.shifts.items():
            start, end = shift_info["hours"]
            if (start <= hour < end) or (shift_name == "Night" and (hour >= 22 or hour < 6)):
                return shift_name, shift_info["label"]
        return "Unknown", "Unknown Shift"

    def get_season(self, month):
        """Determine season based on month"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Spring_Fall"

    def generate_energy_data(self, resolution='15min'):
        """Generate energy data at specified resolution"""
        freq = self.config['time_resolutions'][resolution]
        year = self.data_gen_config['year']
        
        if resolution == '15min':
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq=freq)
        elif resolution == 'hourly':
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq=freq)
        else:  # daily
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq)
        
        data = {"timestamp": dates}
        
        # Add shift and day information
        shift_names = []
        shift_labels = []
        day_names = []
        
        for ts in dates:
            # Day name
            day_names.append(ts.strftime("%A"))
            
            # Shift information (for daily resolution, use midday hour)
            if resolution == 'daily':
                hour = 12  # Use noon as representative hour for daily data
            else:
                hour = ts.hour
                
            shift_name, shift_label = self.get_shift_info(hour)
            shift_names.append(shift_name)
            shift_labels.append(shift_label)
        
        data["shift"] = shift_names
        data["shift_label"] = shift_labels
        data["day_name"] = day_names
        
        # Generate machine data
        for machine, specs in self.machines.items():
            energy = []
            for i, ts in enumerate(dates):
                # --- Check Context ---
                is_weekend = ts.weekday() >= 5
                is_holiday = ts.strftime("%Y-%m-%d") in self.public_holidays
                
                if resolution == 'daily':
                    current_hour = 12  # Use noon as representative
                else:
                    current_hour = ts.hour
                
                # --- Determine Season ---
                season = self.get_season(ts.month)
                
                # --- Shift Logic ---
                active_shift = None
                for shift, shift_info in self.shifts.items():
                    start, end = shift_info["hours"]
                    if (start <= current_hour < end) or (shift == "Night" and (current_hour >= 22 or current_hour < 6)):
                        active_shift = shift
                        break
                
                # --- Energy Calculation ---
                if active_shift and not (is_weekend or is_holiday):
                    # Pick a random operation for the machine
                    op = np.random.choice(specs["operations"])
                    op_kWh = np.random.uniform(*self.operations[op]["kWh_range"])
                    op_kWh *= np.random.normal(1, self.operations[op]["noise"])  # Add noise
                    
                    # Apply seasonal trend
                    seasonal_factor = self.seasonal_trends[season]
                    base_load = specs["base_kWh"] * seasonal_factor["base_load_factor"]
                    op_kWh *= seasonal_factor["operations_factor"]
                    
                    total_kWh = base_load + op_kWh
                    
                    # For 15min resolution, divide hourly consumption by 4
                    if resolution == '15min':
                        total_kWh = total_kWh / 4
                    # For daily resolution, multiply by average daily hours
                    elif resolution == 'daily':
                        total_kWh = total_kWh * self.data_gen_config["daily_operation_hours"]["normal"]
                        
                else:
                    # Weekend/Holiday: Reduced activity
                    activity_factor = np.random.uniform(
                        self.data_gen_config["weekend_activity_factor"]["min"],
                        self.data_gen_config["weekend_activity_factor"]["max"]
                    )
                    base_consumption = specs["base_kWh"] * activity_factor
                    
                    if resolution == '15min':
                        total_kWh = base_consumption / 4
                    elif resolution == 'daily':
                        total_kWh = base_consumption * self.data_gen_config["daily_operation_hours"]["reduced"]
                    else:
                        total_kWh = base_consumption
                
                # --- Simulate Random Failures ---
                if np.random.random() < self.fault_params["failure_rate"]:
                    total_kWh = self.fault_params["failure_kWh"]
                
                energy.append(max(0, total_kWh))  # Ensure non-negative values
            
            data[machine] = np.array(energy)
        
        # Validate generated data
        is_valid, validation_issues = self.validate_data(pd.DataFrame(data), resolution)
        if not is_valid:
            print(f"Warning: Data validation issues found for {resolution} resolution:")
            for issue in validation_issues:
                print(f"  - {issue}")
        
        return pd.DataFrame(data)

    def save_to_hdf5(self, df, resolution='15min'):
        """Save DataFrame to HDF5 file"""
        filename = self.config['file_output']['hdf5_filename_template'].format(
            year=self.data_gen_config['year'], 
            resolution=resolution
        )
        
        with h5py.File(filename, "w") as f:
            # Timestamps (Unix epoch)
            f.create_dataset("timestamps", data=df["timestamp"].astype(np.int64))
            
            # Time metadata
            time_grp = f.create_group("time_info")
            time_grp.create_dataset("shift", data=np.array(df["shift"], dtype='S20'))
            time_grp.create_dataset("shift_label", data=np.array(df["shift_label"], dtype='S50'))
            time_grp.create_dataset("day_name", data=np.array(df["day_name"], dtype='S20'))
            
            # Machine data (group per machine)
            machines_grp = f.create_group("machines")
            for machine in self.machines.keys():
                machines_grp.create_dataset(machine, data=df[machine])
            
            # Metadata (structured groups)
            meta = f.create_group("metadata")
            meta.create_dataset("shifts", data=str(self.shifts).encode('utf-8'))
            meta.create_dataset("holidays", data=np.array(self.public_holidays, dtype='S10'))
            meta.attrs["resolution"] = resolution
            meta.attrs["year"] = self.data_gen_config['year']
            
            # Machine specs (as attributes)
            for machine, specs in self.machines.items():
                machine_grp = machines_grp.create_group(machine + "_specs")
                for k, v in specs.items():
                    if isinstance(v, list):
                        machine_grp.create_dataset(k, data=np.array(v, dtype='S20'))
                    else:
                        machine_grp.attrs[k] = v
            
            # Operations specs
            ops_grp = meta.create_group("operations")
            for op in self.operations.keys():
                op_grp = ops_grp.create_group(op)
                for param, value in self.operations[op].items():
                    if isinstance(value, list):  # kWh_range is now a list in YAML
                        op_grp.create_dataset(param, data=np.array(value))
                    else:
                        op_grp.attrs[param] = value
        
        print(f"Data saved to {filename}")
        
        # Generate and save metadata
        metadata = self.generate_metadata(df, resolution)
        
        # Add validation results to metadata
        is_valid, validation_issues = self.validate_data(df, resolution)
        metadata["data_quality"]["validation_passed"] = is_valid
        metadata["data_quality"]["validation_issues"] = validation_issues
        
        self.save_metadata(metadata, resolution)

# ====================== VISUALIZATION CLASS ======================
class EnergyVisualizer:
    def __init__(self, config):
        """Initialize visualizer with configuration"""
        self.config = config
        self.viz_config = config['visualization']
        self.file_config = config['file_output']
        
    def plot_energy_trends_all_resolutions(self, df_15min, df_hourly, df_daily, 
                                         machine=None, days=None):
        """Create comprehensive plots for all time resolutions"""
        if machine is None:
            machine = self.viz_config['default_machine']
        if days is None:
            days = self.viz_config['default_days_to_plot']
            
        fig, axes = plt.subplots(3, 1, figsize=(
            self.viz_config['figure_size']['width'], 
            self.viz_config['figure_size']['height']
        ))
        
        colors = self.viz_config['colors']
        
        # 15-minute resolution plot
        end_idx = days * 24 * 4  # 4 points per hour
        axes[0].plot(df_15min["timestamp"][:end_idx], df_15min[machine][:end_idx], 
                     label=f"{machine} - 15min", color=colors['resolution_15min'], alpha=0.7)
        axes[0].set_title(f"15-Minute Resolution: {machine} Energy Consumption (First {days} Days)")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Hourly resolution plot
        end_idx_hourly = days * 24
        axes[1].plot(df_hourly["timestamp"][:end_idx_hourly], df_hourly[machine][:end_idx_hourly], 
                     label=f"{machine} - Hourly", color=colors['resolution_hourly'], alpha=0.8)
        axes[1].set_title(f"Hourly Resolution: {machine} Energy Consumption (First {days} Days)")
        axes[1].set_ylabel("Energy (kWh)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Daily resolution plot
        end_idx_daily = days
        axes[2].plot(df_daily["timestamp"][:end_idx_daily], df_daily[machine][:end_idx_daily], 
                     label=f"{machine} - Daily", color=colors['resolution_daily'], marker='o', alpha=0.8)
        axes[2].set_title(f"Daily Resolution: {machine} Energy Consumption (First {days} Days)")
        axes[2].set_ylabel("Energy (kWh)")
        axes[2].set_xlabel("Time")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        filename = self.file_config['plot_filename_template'].format(
            machine=machine, plot_type=f"all_resolutions_{days}days", resolution="combined"
        )
        plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
                   bbox_inches=self.file_config['bbox_inches'])
        plt.show()

    def plot_yearly_daily_trend(self, df_daily, machine=None):
        """Plot full year daily energy consumption trend"""
        if machine is None:
            machine = self.viz_config['default_machine']
        
        plt.figure(figsize=(
            self.viz_config['figure_size']['yearly_width'], 
            self.viz_config['figure_size']['yearly_height']
        ))
        
        # Main plot
        plt.plot(df_daily["timestamp"], df_daily[machine], 
                color=self.viz_config['colors']['yearly_daily'], 
                linewidth=1.5, alpha=0.8, label=f'{machine} Daily Consumption')
        
        # Add moving average
        window = 7  # 7-day moving average
        df_daily[f'{machine}_ma'] = df_daily[machine].rolling(window=window, center=True).mean()
        plt.plot(df_daily["timestamp"], df_daily[f'{machine}_ma'], 
                color='red', linewidth=2, alpha=0.9, 
                label=f'{window}-Day Moving Average')
        
        # Highlight weekends
        weekend_mask = df_daily['timestamp'].dt.weekday >= 5
        weekend_data = df_daily[weekend_mask]
        plt.scatter(weekend_data["timestamp"], weekend_data[machine], 
                   color='orange', alpha=0.6, s=15, label='Weekends', zorder=3)
        
        # Highlight holidays
        holiday_mask = df_daily['timestamp'].dt.strftime('%Y-%m-%d').isin(
            self.config['public_holidays']
        )
        holiday_data = df_daily[holiday_mask]
        if not holiday_data.empty:
            plt.scatter(holiday_data["timestamp"], holiday_data[machine], 
                       color='red', alpha=0.8, s=30, marker='x', 
                       label='Public Holidays', zorder=4)
        
        # Seasonal backgrounds
        for i, (season, color) in enumerate([
            ('Winter', '#E6F3FF'), ('Spring', '#E6FFE6'), 
            ('Summer', '#FFFFE6'), ('Fall', '#FFE6E6')
        ]):
            if season == 'Winter':
                # Winter spans year boundary
                start1, end1 = '2024-01-01', '2024-02-29'
                start2, end2 = '2024-12-01', '2024-12-31'
                plt.axvspan(pd.to_datetime(start1), pd.to_datetime(end1), 
                           alpha=0.1, color=color, label=season if i == 0 else "")
                plt.axvspan(pd.to_datetime(start2), pd.to_datetime(end2), 
                           alpha=0.1, color=color)
            elif season == 'Spring':
                plt.axvspan(pd.to_datetime('2024-03-01'), pd.to_datetime('2024-05-31'), 
                           alpha=0.1, color=color, label=season)
            elif season == 'Summer':
                plt.axvspan(pd.to_datetime('2024-06-01'), pd.to_datetime('2024-08-31'), 
                           alpha=0.1, color=color, label=season)
            else:  # Fall
                plt.axvspan(pd.to_datetime('2024-09-01'), pd.to_datetime('2024-11-30'), 
                           alpha=0.1, color=color, label=season)
        
        plt.title(f'{machine} - Full Year Daily Energy Consumption (2024)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy Consumption (kWh)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        plt.xticks(rotation=45)
        months = pd.date_range('2024-01-01', '2024-12-31', freq='MS')
        plt.xticks(months, [month.strftime('%b') for month in months])
        
        plt.tight_layout()
        filename = self.file_config['plot_filename_template'].format(
            machine=machine, plot_type="yearly_daily_trend", resolution="daily"
        )
        plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
                   bbox_inches=self.file_config['bbox_inches'])
        plt.show()
        
        # Print yearly statistics
        print(f"\n=== YEARLY STATISTICS FOR {machine} ===")
        print(f"Total Annual Consumption: {df_daily[machine].sum():.2f} kWh")
        print(f"Average Daily Consumption: {df_daily[machine].mean():.2f} kWh")
        print(f"Peak Daily Consumption: {df_daily[machine].max():.2f} kWh on {df_daily.loc[df_daily[machine].idxmax(), 'timestamp'].strftime('%Y-%m-%d')}")
        print(f"Minimum Daily Consumption: {df_daily[machine].min():.2f} kWh on {df_daily.loc[df_daily[machine].idxmin(), 'timestamp'].strftime('%Y-%m-%d')}")
        
        # Seasonal averages
        df_temp = df_daily.copy()
        df_temp['season'] = df_temp['timestamp'].dt.month.apply(
            lambda x: 'Winter' if x in [12, 1, 2] else 
                     'Spring' if x in [3, 4, 5] else 
                     'Summer' if x in [6, 7, 8] else 'Fall'
        )
        seasonal_avg = df_temp.groupby('season')[machine].mean()
        print(f"Seasonal Averages:")
        for season, avg in seasonal_avg.items():
            print(f"  {season}: {avg:.2f} kWh/day")

    def generate_data_quality_report(self, df_dict):
        """Generate comprehensive data quality report"""
        print("\n" + "="*60)
        print("           DATA QUALITY REPORT")
        print("="*60)
        
        for resolution, df in df_dict.items():
            print(f"\n{resolution.upper()} RESOLUTION:")
            print("-" * 30)
            
            # Basic statistics
            print(f"Records: {len(df):,}")
            print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Time Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
            
            # Data completeness
            missing_data = df.isnull().sum().sum()
            print(f"Missing Values: {missing_data}")
            
            # Energy statistics
            energy_cols = [col for col in df.columns if col not in ['timestamp', 'shift', 'shift_label', 'day_name']]
            total_energy = df[energy_cols].sum().sum()
            print(f"Total Energy Consumption: {total_energy:.2f} kWh")
            
            # Validation summary
            is_valid, issues = self.generator.validate_data(df, resolution) if hasattr(self, 'generator') else (True, [])
            print(f"Validation Status: {'PASSED' if is_valid else 'FAILED'}")
            if issues:
                print("Issues Found:")
                for issue in issues[:5]:  # Show first 5 issues
                    print(f"  - {issue}")
                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more issues")
        
        print("\n" + "="*60)

    def plot_shift_analysis(self, df, machine=None, resolution='hourly'):
        """Plot energy consumption by shift"""
        if machine is None:
            machine = self.viz_config['default_machine']
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot by shift
        df_sample = df.head(24*7)  # First week
        sns.boxplot(data=df_sample, x='shift', y=machine, ax=ax1)
        ax1.set_title(f'{machine} Energy by Shift ({resolution})')
        ax1.set_ylabel('Energy (kWh)')
        
        # Average energy by day of week
        df_sample['day_name'] = pd.Categorical(df_sample['day_name'], 
                                              categories=['Monday', 'Tuesday', 'Wednesday', 
                                                        'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                              ordered=True)
        day_avg = df_sample.groupby('day_name')[machine].mean()
        
        ax2.bar(day_avg.index, day_avg.values, color='skyblue', alpha=0.7)
        ax2.set_title(f'{machine} Average Energy by Day ({resolution})')
        ax2.set_ylabel('Average Energy (kWh)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = self.file_config['plot_filename_template'].format(
            machine=machine, plot_type="shift_analysis", resolution=resolution
        )
        plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
                   bbox_inches=self.file_config['bbox_inches'])
        plt.show()

    def plot_weekly_heatmap(self, df, machine=None, resolution='hourly'):
        """Create a weekly heatmap showing energy patterns"""
        if machine is None:
            machine = self.viz_config['default_machine']
            
        # Take first 4 weeks for better visualization
        df_sample = df.head(24*7*4 if resolution == 'hourly' else 
                           (24*4*7*4 if resolution == '15min' else 28))
        
        if resolution == 'hourly':
            df_sample['hour'] = df_sample['timestamp'].dt.hour
            pivot_data = df_sample.groupby(['day_name', 'hour'])[machine].mean().unstack()
        elif resolution == '15min':
            df_sample['hour_15min'] = df_sample['timestamp'].dt.hour + df_sample['timestamp'].dt.minute/60
            # Group into hour bins for visualization
            df_sample['hour'] = df_sample['timestamp'].dt.hour
            pivot_data = df_sample.groupby(['day_name', 'hour'])[machine].mean().unstack()
        else:  # daily
            pivot_data = df_sample.groupby('day_name')[machine].mean().to_frame().T
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(day_order)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=False, cmap=self.viz_config['heatmap_colormap'], 
                   cbar_kws={'label': 'Energy (kWh)'})
        plt.title(f'{machine} Energy Consumption Heatmap - {resolution.title()} Resolution')
        plt.ylabel('Day of Week')
        if resolution != 'daily':
            plt.xlabel('Hour of Day')
        plt.tight_layout()
        filename = self.file_config['plot_filename_template'].format(
            machine=machine, plot_type="heatmap", resolution=resolution
        )
        plt.savefig(filename, dpi=self.viz_config['plot_dpi'], 
                   bbox_inches=self.file_config['bbox_inches'])
        plt.show()

# ====================== MAIN EXECUTION ======================
def main():
    """Main execution function"""
    print("=== Energy Data Generator with YAML Configuration ===")
    
    # Initialize generator and visualizer
    generator = EnergyDataGenerator("config.yaml")
    visualizer = EnergyVisualizer(generator.config)
    visualizer.generator = generator  # For validation access
    
    print("Generating advanced energy datasets for all resolutions...")
    
    # Generate data for all resolutions
    print("Generating 15-minute resolution data...")
    df_15min = generator.generate_energy_data('15min')
    print("Generating hourly resolution data...")
    df_hourly = generator.generate_energy_data('hourly')
    print("Generating daily resolution data...")
    df_daily = generator.generate_energy_data('daily')
    
    print(f"\n15-minute data shape: {df_15min.shape}")
    print(f"Hourly data shape: {df_hourly.shape}")
    print(f"Daily data shape: {df_daily.shape}")
    
    # Save to HDF5 (includes metadata export)
    print("\nSaving datasets and metadata...")
    generator.save_to_hdf5(df_15min, '15min')
    generator.save_to_hdf5(df_hourly, 'hourly')
    generator.save_to_hdf5(df_daily, 'daily')
    print("All datasets and metadata saved.")
    
    # Data quality report
    if generator.config.get('validation', {}).get('data_quality_report', True):
        df_dict = {'15min': df_15min, 'hourly': df_hourly, 'daily': df_daily}
        visualizer.generate_data_quality_report(df_dict)
    
    # Display sample data
    print("\nSample 15-minute data:")
    default_machine = generator.config['visualization']['default_machine']
    print(df_15min[['timestamp', 'shift', 'day_name', default_machine]].head(10))
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # 1. Multi-resolution comparison
    visualizer.plot_energy_trends_all_resolutions(df_15min, df_hourly, df_daily)
    
    # 2. Full year daily trend plot
    if generator.config['visualization'].get('yearly_plot_enabled', True):
        print("Generating yearly daily trend plot...")
        visualizer.plot_yearly_daily_trend(df_daily)
    
    # 3. Shift analysis for each resolution
    for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
        visualizer.plot_shift_analysis(df, resolution=resolution)
        
    # 4. Weekly heatmaps
    for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
        visualizer.plot_weekly_heatmap(df, resolution=resolution)
    
    print("All visualizations complete!")
    
    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for resolution, df in [('15min', df_15min), ('hourly', df_hourly), ('daily', df_daily)]:
        print(f"\n{resolution.upper()} RESOLUTION:")
        print(f"Data points: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"{default_machine} average consumption: {df[default_machine].mean():.2f} kWh")
        print(f"Shift distribution:")
        print(df['shift'].value_counts())

if __name__ == "__main__":
    main()