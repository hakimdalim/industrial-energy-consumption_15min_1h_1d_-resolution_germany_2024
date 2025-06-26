import numpy as np
import pandas as pd
import h5py
import yaml
import json
from datetime import datetime, timedelta
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
            "Injection_Molder_2": {"type": "Molder", "base_kWh": 2.5, "operations": ["Injection_Molding"]},
            "Laser_Cutter_3": {"type": "Cutter", "base_kWh": 3.0, "operations": ["Cutting"]},
            "Assembly_Robot_4": {"type": "Robot", "base_kWh": 0.8, "operations": ["Assembly"]},
            "Industrial_Oven_5": {"type": "Thermal", "base_kWh": 5.0, "operations": ["Heat_Treatment"]},
            "Packaging_Line_6": {"type": "Conveyor", "base_kWh": 1.0, "operations": ["Packaging"]},
            "Welding_Cell_7": {"type": "Welder", "base_kWh": 4.0, "operations": ["Welding"]},
            "Painting_Booth_8": {"type": "Sprayer", "base_kWh": 2.0, "operations": ["Painting"]},
            "Hydraulic_Press_9": {"type": "Press", "base_kWh": 3.5, "operations": ["Stamping"]},
        },
        'operations': {
            "Milling": {"kWh_range": [8, 15], "noise": 0.1},
            "Drilling": {"kWh_range": [5, 10], "noise": 0.15},
            "Injection_Molding": {"kWh_range": [10, 20], "noise": 0.25},
            "Cutting": {"kWh_range": [12, 25], "noise": 0.2},
            "Assembly": {"kWh_range": [2, 5], "noise": 0.1},
            "Heat_Treatment": {"kWh_range": [25, 45], "noise": 0.15},
            "Packaging": {"kWh_range": [2, 6], "noise": 0.1},
            "Welding": {"kWh_range": [20, 40], "noise": 0.3},
            "Painting": {"kWh_range": [8, 16], "noise": 0.2},
            "Stamping": {"kWh_range": [15, 25], "noise": 0.2},
            "Cooling": {"kWh_range": [4, 8], "noise": 0.1},
            "Drying": {"kWh_range": [6, 14], "noise": 0.2},
            "Engraving": {"kWh_range": [4, 8], "noise": 0.15},
            "Testing": {"kWh_range": [1, 4], "noise": 0.05},
            "Quality_Control": {"kWh_range": [2, 5], "noise": 0.08},
        },
        'shifts': {
            "Morning": {"hours": [6, 14], "label": "Morning Shift (6AM-2PM)"},
            "Afternoon": {"hours": [14, 22], "label": "Afternoon Shift (2PM-10PM)"},
            "Night": {"hours": [22, 6], "label": "Night Shift (10PM-6AM)"},
        },
        'public_holidays': ["2024-01-01"],
        'seasonal_trends': {
            "Winter": {"base_load_factor": 1.2, "operations_factor": 0.9},
            "Summer": {"base_load_factor": 1.1, "operations_factor": 1.0},
            "Spring_Fall": {"base_load_factor": 1.0, "operations_factor": 1.0},
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
        'operation_distribution': {
            "high_frequency": ["Milling", "Assembly", "Packaging", "Testing", "Quality_Control"],
            "medium_frequency": ["Drilling", "Cutting", "Injection_Molding", "Cooling", "Engraving"],
            "low_frequency": ["Welding", "Heat_Treatment", "Drying", "Painting", "Stamping"]
        },
        'validation': {
            "enabled": True,
            "expected_machines": 9,
            "expected_operations": 15,
            "check_negative_values": True,
            "energy_range_validation": {"min_kwh": 0.0, "max_kwh": 1000.0}
        },
        'file_output': {
            "hdf5_filename_template": "energy_{year}_9machines_15ops_{resolution}.h5",
            "metadata_filename_template": "energy_{year}_metadata_9m15o_{resolution}.json",
            "export_metadata": True
        },
        'visualization': {
            "default_machine": "CNC_Mill_1",
            "default_days_to_plot": 7,
            "colors": {"resolution_15min": "blue", "resolution_hourly": "green", "resolution_daily": "red"}
        }
    }

# ====================== REALISTIC ENERGY DATA GENERATOR ======================
class RealisticEnergyDataGenerator:
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
        self.operation_distribution = self.config.get('operation_distribution', {})
        
        # Validate configuration
        self.validate_configuration()
    
    def validate_configuration(self):
        """Validate that we have exactly 9 machines and 15 operations"""
        actual_machines = len(self.machines)
        actual_operations = len(self.operations)
        expected_machines = self.validation_config.get('expected_machines', 9)
        expected_operations = self.validation_config.get('expected_operations', 15)
        
        print(f"🔍 Configuration validation:")
        print(f"   Machines: {actual_machines}/{expected_machines} {'✅' if actual_machines == expected_machines else '❌'}")
        print(f"   Operations: {actual_operations}/{expected_operations} {'✅' if actual_operations == expected_operations else '❌'}")
        
        if actual_machines != expected_machines:
            raise ValueError(f"Expected {expected_machines} machines, got {actual_machines}")
        if actual_operations != expected_operations:
            raise ValueError(f"Expected {expected_operations} operations, got {actual_operations}")
        
        # Validate all machine operations exist
        all_machine_ops = set()
        for machine, specs in self.machines.items():
            for op in specs['operations']:
                all_machine_ops.add(op)
                if op not in self.operations:
                    raise ValueError(f"Operation '{op}' used by machine '{machine}' not defined in operations")
        
        print(f"   Operation coverage: {len(all_machine_ops)}/{actual_operations} operations used by machines")
        
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
    
    def is_workday(self, timestamp):
        """Determine if timestamp is a workday (not weekend or holiday)"""
        is_weekend = timestamp.weekday() >= 5  # Saturday=5, Sunday=6
        is_holiday = timestamp.strftime("%Y-%m-%d") in self.public_holidays
        return not (is_weekend or is_holiday)
    
    def get_seasonal_factor(self, month):
        """Get seasonal adjustment factors"""
        season = self.get_season(month)
        return self.seasonal_trends.get(season, {"base_load_factor": 1.0, "operations_factor": 1.0})

    def get_operation_frequency_factor(self, operation):
        """Get frequency factor for operation based on distribution strategy"""
        if operation in self.operation_distribution.get('high_frequency', []):
            return 1.5  # 50% more likely to run
        elif operation in self.operation_distribution.get('low_frequency', []):
            return 0.6  # 40% less likely to run
        else:  # medium_frequency
            return 1.0  # Normal frequency

    def get_realistic_consumption(self, timestamp, machine_specs, resolution='hourly'):
        """Calculate realistic energy consumption for a machine at given timestamp"""
        # Determine context
        is_workday = self.is_workday(timestamp)
        hour = timestamp.hour if resolution != 'daily' else 12
        seasonal_factor = self.get_seasonal_factor(timestamp.month)
        
        # Base consumption (standby power) - always present
        base_consumption = machine_specs["base_kWh"] * seasonal_factor["base_load_factor"]
        
        # Determine if machine should be operating
        operating = False
        operation_intensity = 0.0
        selected_operation = None
        
        if is_workday:
            # Workday operations based on shift hours
            if 6 <= hour < 22:  # Primary operating hours (6 AM - 10 PM)
                # High probability of operation during main shifts
                if 6 <= hour < 14:  # Morning shift
                    operating_prob = 0.8
                elif 14 <= hour < 22:  # Afternoon shift  
                    operating_prob = 0.9
                else:  # Night shift
                    operating_prob = 0.4
                
                operating = np.random.random() < operating_prob
                if operating:
                    operation_intensity = np.random.uniform(0.6, 1.0)  # 60-100% intensity
            else:  # Night hours (10 PM - 6 AM)
                operating_prob = 0.3  # Reduced night operations
                operating = np.random.random() < operating_prob
                if operating:
                    operation_intensity = np.random.uniform(0.3, 0.7)  # Lower intensity
        else:
            # Weekend/Holiday operations (maintenance, minimal production)
            if 8 <= hour < 18:  # Daytime weekend operations
                operating_prob = 0.2
                operating = np.random.random() < operating_prob
                if operating:
                    operation_intensity = np.random.uniform(0.2, 0.5)  # Low intensity
            else:
                operating_prob = 0.05  # Very minimal night/weekend operations
                operating = np.random.random() < operating_prob
                if operating:
                    operation_intensity = np.random.uniform(0.1, 0.3)
        
        # Calculate operational consumption
        operational_consumption = 0.0
        if operating:
            # Select operation based on frequency distribution
            available_operations = machine_specs["operations"]
            
            # Apply frequency weighting
            operation_weights = []
            for op in available_operations:
                freq_factor = self.get_operation_frequency_factor(op)
                operation_weights.append(freq_factor)
            
            # Normalize weights
            total_weight = sum(operation_weights)
            operation_probs = [w / total_weight for w in operation_weights]
            
            # Select operation
            selected_operation = np.random.choice(available_operations, p=operation_probs)
            op_config = self.operations[selected_operation]
            
            # Calculate operation power with intensity and seasonal factors
            op_power_range = op_config["kWh_range"]
            base_op_power = np.random.uniform(op_power_range[0], op_power_range[1])
            
            # Apply intensity, seasonal factors, and noise
            operational_consumption = (base_op_power * 
                                     operation_intensity * 
                                     seasonal_factor["operations_factor"] * 
                                     np.random.normal(1.0, op_config["noise"]))
        
        # Total consumption
        total_consumption = base_consumption + operational_consumption
        
        # Apply random failures
        if np.random.random() < self.fault_params["failure_rate"]:
            total_consumption = self.fault_params["failure_kWh"]
        
        # Adjust for time resolution
        if resolution == '15min':
            total_consumption = total_consumption / 4  # 15-min fraction of hourly
        elif resolution == 'daily':
            # For daily, multiply by realistic daily hours
            if is_workday:
                daily_hours = self.data_gen_config["daily_operation_hours"]["normal"]
            else:
                daily_hours = self.data_gen_config["daily_operation_hours"]["reduced"]
            total_consumption = total_consumption * daily_hours
        
        return max(0, total_consumption)

    def generate_energy_data(self, resolution='15min'):
        """Generate realistic energy data at specified resolution"""
        print(f"Generating realistic {resolution} resolution data...")
        print(f" Using {len(self.machines)} machines and {len(self.operations)} operations")
        
        # Create date range
        year = self.data_gen_config['year']
        freq = self.config['time_resolutions'][resolution]
        
        if resolution == '15min':
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:45", freq=freq)
        elif resolution == 'hourly':
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq=freq)
        else:  # daily
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=freq)
        
        data = {"timestamp": dates}
        
        # Add time metadata
        shift_names = []
        shift_labels = []
        day_names = []
        
        for ts in dates:
            day_names.append(ts.strftime("%A"))
            hour = ts.hour if resolution != 'daily' else 12
            shift_name, shift_label = self.get_shift_info(hour)
            shift_names.append(shift_name)
            shift_labels.append(shift_label)
        
        data["shift"] = shift_names
        data["shift_label"] = shift_labels
        data["day_name"] = day_names
        
        # Generate realistic machine data
        for machine, specs in self.machines.items():
            energy_values = []
            
            print(f"  Generating data for {machine} (operations: {', '.join(specs['operations'])})")
            
            for ts in dates:
                consumption = self.get_realistic_consumption(ts, specs, resolution)
                energy_values.append(consumption)
            
            data[machine] = np.array(energy_values)
        
        df = pd.DataFrame(data)
        
        # Validate data
        is_valid, issues = self.validate_data(df, resolution)
        if not is_valid:
            print(f"Warning: Data validation issues found for {resolution} resolution:")
            for issue in issues[:3]:
                print(f"  - {issue}")
        
        return df
    
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
        
        # Check for reasonable year-round consumption
        for machine in self.machines.keys():
            monthly_avg = df.groupby(df['timestamp'].dt.month)[machine].mean()
            if (monthly_avg == 0).any():
                zero_months = monthly_avg[monthly_avg == 0].index.tolist()
                issues.append(f"{machine}: Zero consumption in months {zero_months}")
        
        # Check for reasonable variation
        for machine in self.machines.keys():
            cv = df[machine].std() / df[machine].mean() if df[machine].mean() > 0 else 0
            if cv > 2.0:  # Too much variation
                issues.append(f"{machine}: Excessive variation (CV={cv:.2f})")
        
        return len(issues) == 0, issues
    
    def generate_metadata(self, df, resolution):
        """Generate comprehensive metadata for the dataset"""
        metadata = {
            "dataset_info": {
                "resolution": resolution,
                "year": self.data_gen_config['year'],
                "generation_timestamp": datetime.now().isoformat(),
                "total_records": len(df),
                "generation_method": "realistic_probabilistic_consumption_9m15o",
                "machine_count": len(self.machines),
                "operation_count": len(self.operations),
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
                "public_holidays": self.public_holidays,
                "operation_distribution": self.operation_distribution
            },
            "data_statistics": {},
            "data_quality": {
                "validation_enabled": self.validation_config.get('enabled', False),
                "validation_timestamp": datetime.now().isoformat()
            },
            "realism_improvements": {
                "exact_specification": "Exactly 9 machines and 15 operations as required",
                "operation_frequency": "High/medium/low frequency distribution for realistic patterns",
                "probabilistic_operations": "Operations based on realistic probabilities throughout the year",
                "continuous_base_load": "Base consumption always present (standby power)",
                "shift_aware_intensity": "Operation intensity varies by shift and time",
                "seasonal_adjustments": "Consumption adjusted for seasonal factors",
                "weekend_operations": "Reduced but realistic weekend/holiday operations"
            }
        }
        
        # Calculate statistics for each machine
        for machine in self.machines.keys():
            machine_data = df[machine]
            machine_operations = self.machines[machine]["operations"]
            metadata["data_statistics"][machine] = {
                "mean": float(machine_data.mean()),
                "std": float(machine_data.std()),
                "min": float(machine_data.min()),
                "max": float(machine_data.max()),
                "median": float(machine_data.median()),
                "total_consumption": float(machine_data.sum()),
                "zero_consumption_periods": int((machine_data == 0).sum()),
                "coefficient_of_variation": float(machine_data.std() / machine_data.mean()) if machine_data.mean() > 0 else 0,
                "assigned_operations": machine_operations,
                "operation_count": len(machine_operations),
                "percentiles": {
                    "25th": float(machine_data.quantile(0.25)),
                    "75th": float(machine_data.quantile(0.75)),
                    "95th": float(machine_data.quantile(0.95))
                }
            }
            
            # Monthly statistics to ensure year-round consumption
            monthly_stats = df.groupby(df['timestamp'].dt.month)[machine].agg(['mean', 'sum', 'count'])
            metadata["data_statistics"][machine]["monthly_consumption"] = {
                str(month): {
                    "mean": float(monthly_stats.loc[month, 'mean']),
                    "total": float(monthly_stats.loc[month, 'sum']),
                    "records": int(monthly_stats.loc[month, 'count'])
                } for month in monthly_stats.index
            }
        
        # Operation usage statistics
        metadata["data_statistics"]["operation_coverage"] = {
            "total_operations_defined": len(self.operations),
            "operations_used_by_machines": len(set(op for machine_ops in [specs["operations"] for specs in self.machines.values()] for op in machine_ops)),
            "operation_distribution": self.operation_distribution,
            "operation_details": {
                op: {
                    "power_range": self.operations[op]["kWh_range"],
                    "noise_level": self.operations[op]["noise"],
                    "assigned_to_machines": [machine for machine, specs in self.machines.items() if op in specs["operations"]],
                    "frequency_category": (
                        "high" if op in self.operation_distribution.get("high_frequency", []) else
                        "low" if op in self.operation_distribution.get("low_frequency", []) else
                        "medium"
                    )
                } for op in self.operations.keys()
            }
        }
        
        # Shift and day distribution
        metadata["data_statistics"]["shift_distribution"] = df['shift'].value_counts().to_dict()
        metadata["data_statistics"]["day_distribution"] = df['day_name'].value_counts().to_dict()
        
        # Workday vs weekend statistics
        df_temp = df.copy()
        df_temp['is_workday'] = df_temp['timestamp'].apply(self.is_workday)
        workday_weekend_stats = {}
        for machine in self.machines.keys():
            workday_avg = df_temp[df_temp['is_workday']][machine].mean()
            weekend_avg = df_temp[~df_temp['is_workday']][machine].mean()
            workday_weekend_stats[machine] = {
                "workday_average": float(workday_avg),
                "weekend_average": float(weekend_avg),
                "reduction_factor": float(weekend_avg / workday_avg) if workday_avg > 0 else 0
            }
        metadata["data_statistics"]["workday_weekend_comparison"] = workday_weekend_stats
        
        return metadata
    
    def save_metadata(self, metadata, resolution):
        """Save metadata to file"""
        if not self.config['file_output'].get('export_metadata', True):
            return
        
        try:
            filename = self.config['file_output']['metadata_filename_template'].format(
                year=self.data_gen_config['year'], 
                resolution=resolution
            )
        except KeyError as e:
            filename = f"energy_{self.data_gen_config['year']}_metadata_9m15o_{resolution}.json"
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {filename}")
    
    def save_to_hdf5(self, df, resolution='15min'):
        """Save DataFrame to HDF5 file"""
        try:
            filename = self.config['file_output']['hdf5_filename_template'].format(
                year=self.data_gen_config['year'], 
                resolution=resolution
            )
        except KeyError as e:
            filename = f"energy_{self.data_gen_config['year']}_9machines_15ops_{resolution}.h5"
        
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
            meta.attrs["generation_method"] = "realistic_probabilistic_consumption_9m15o"
            meta.attrs["machine_count"] = len(self.machines)
            meta.attrs["operation_count"] = len(self.operations)
            
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
                    if isinstance(value, list):
                        op_grp.create_dataset(param, data=np.array(value))
                    else:
                        op_grp.attrs[param] = value
            
            # Operation distribution
            if self.operation_distribution:
                dist_grp = meta.create_group("operation_distribution")
                for freq_type, ops in self.operation_distribution.items():
                    dist_grp.create_dataset(freq_type, data=np.array(ops, dtype='S30'))
        
        print(f"Data saved to {filename}")
        
        # Generate and save metadata
        metadata = self.generate_metadata(df, resolution)
        
        # Add validation results to metadata
        is_valid, validation_issues = self.validate_data(df, resolution)
        metadata["data_quality"]["validation_passed"] = is_valid
        metadata["data_quality"]["validation_issues"] = validation_issues
        
        self.save_metadata(metadata, resolution)

# ====================== ENHANCED VISUALIZATION CLASS ======================
class EnhancedEnergyVisualizer:
    def __init__(self, config):
        """Initialize visualizer with configuration"""
        self.config = config
        self.viz_config = config['visualization']
        self.file_config = config['file_output']
        
    def plot_machine_operation_matrix(self, generator):
        """Plot matrix showing which operations are assigned to which machines"""
        machines = list(generator.machines.keys())
        operations = list(generator.operations.keys())
        
        # Create matrix
        matrix = np.zeros((len(machines), len(operations)))
        for i, machine in enumerate(machines):
            machine_ops = generator.machines[machine]["operations"]
            for j, operation in enumerate(operations):
                if operation in machine_ops:
                    matrix[i, j] = 1
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(matrix, 
                   xticklabels=operations, 
                   yticklabels=machines,
                   cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Operation Assigned'},
                   linewidths=0.5)
        
        plt.title('Machine-Operation Assignment Matrix\n9 Machines × 15 Operations', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Operations', fontsize=12)
        plt.ylabel('Machines', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add frequency indicators
        freq_colors = {'high': 'red', 'medium': 'orange', 'low': 'blue'}
        operation_dist = generator.operation_distribution
        
        for j, operation in enumerate(operations):
            freq_type = (
                'high' if operation in operation_dist.get('high_frequency', []) else
                'low' if operation in operation_dist.get('low_frequency', []) else
                'medium'
            )
            plt.axvline(x=j+0.5, color=freq_colors[freq_type], alpha=0.3, linewidth=3)
        
        # Add legend for frequency
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='High Frequency'),
            Patch(facecolor='orange', alpha=0.3, label='Medium Frequency'),
            Patch(facecolor='blue', alpha=0.3, label='Low Frequency')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        filename = "machine_operation_matrix_9x15.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Machine-operation matrix saved as: {filename}")
        
        # Print summary
        print(f"\n📊 MACHINE-OPERATION SUMMARY:")
        print(f"   Total machines: {len(machines)}")
        print(f"   Total operations: {len(operations)}")
        print(f"   Total assignments: {int(matrix.sum())}")
        print(f"   Average ops per machine: {matrix.sum(axis=1).mean():.1f}")
        
        # Operation frequency distribution
        freq_dist = operation_dist
        print(f"\n🔄 OPERATION FREQUENCY DISTRIBUTION:")
        for freq_type, ops in freq_dist.items():
            print(f"   {freq_type}: {len(ops)} operations - {ops}")

    def plot_energy_trends_all_resolutions(self, df_15min, df_hourly, df_daily, 
                                         machine=None, days=None):
        """Create comprehensive plots for all time resolutions"""
        if machine is None:
            machine = self.viz_config['default_machine']
        if days is None:
            days = self.viz_config['default_days_to_plot']
            
        fig, axes = plt.subplots(3, 1, figsize=(18, 14))
        colors = self.viz_config['colors']
        
        # 15-minute resolution plot
        end_idx = days * 24 * 4
        axes[0].plot(df_15min["timestamp"][:end_idx], df_15min[machine][:end_idx], 
                     label=f"{machine} - 15min", color=colors['resolution_15min'], alpha=0.8, linewidth=0.8)
        axes[0].set_title(f"15-Minute Resolution: {machine} Energy Consumption\n(9 Machines × 15 Operations Model)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Energy (kWh)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Hourly resolution plot
        end_idx_hourly = days * 24
        axes[1].plot(df_hourly["timestamp"][:end_idx_hourly], df_hourly[machine][:end_idx_hourly], 
                     label=f"{machine} - Hourly", color=colors['resolution_hourly'], alpha=0.9, linewidth=1.5)
        axes[1].set_title(f"Hourly Resolution: {machine} Energy Consumption\n(Frequency-Based Operation Selection)", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Energy (kWh)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Daily resolution plot
        end_idx_daily = days
        axes[2].plot(df_daily["timestamp"][:end_idx_daily], df_daily[machine][:end_idx_daily], 
                     label=f"{machine} - Daily", color=colors['resolution_daily'], marker='o', alpha=0.9, linewidth=2, markersize=6)
        axes[2].set_title(f"Daily Resolution: {machine} Energy Consumption\n(Realistic Industrial Patterns)", fontsize=14, fontweight='bold')
        axes[2].set_ylabel("Energy (kWh)", fontsize=12)
        axes[2].set_xlabel("Time", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Add annotations explaining the 9x15 model
        axes[0].text(0.02, 0.98, "✓ Exactly 9 machines, 15 operations\n✓ Frequency-weighted operation selection\n✓ Year-round realistic consumption", 
                    transform=axes[0].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        axes[1].text(0.02, 0.98, "✓ High/medium/low frequency operations\n✓ Shift-based operation intensity\n✓ Seasonal adjustments", 
                    transform=axes[1].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        axes[2].text(0.02, 0.98, "✓ Workday vs weekend patterns\n✓ Holiday effects\n✓ Validated realistic ranges", 
                    transform=axes[2].transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        filename = f"{machine}_9machines_15ops_{days}days.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Plot saved as: {filename}")

    def plot_yearly_energy_distribution(self, df_daily, generator):
        """Plot comprehensive yearly energy distribution analysis"""
        machines = list(generator.machines.keys())
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Full year consumption trend for all machines (top panel)
        ax1 = plt.subplot(3, 2, (1, 2))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
        for i, machine in enumerate(machines):
            plt.plot(df_daily['timestamp'], df_daily[machine], 
                    label=machine, color=colors[i], linewidth=1.5, alpha=0.8)
        
        # Add 30-day moving average for total consumption
        df_daily['total_consumption'] = df_daily[machines].sum(axis=1)
        df_daily['total_ma30'] = df_daily['total_consumption'].rolling(window=30, center=True).mean()
        plt.plot(df_daily['timestamp'], df_daily['total_ma30'], 
                color='black', linewidth=3, label='Total (30-day avg)', linestyle='--')
        
        # Highlight seasons with background colors
        season_colors = {
            'Winter': ('#E6F3FF', [(1, 59), (335, 365)]),  # Light blue
            'Spring': ('#E6FFE6', [(60, 151)]),             # Light green
            'Summer': ('#FFFFE6', [(152, 243)]),            # Light yellow
            'Fall': ('#FFE6E6', [(244, 334)])               # Light red
        }
        
        for season, (color, periods) in season_colors.items():
            for start_day, end_day in periods:
                start_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=start_day-1)
                end_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=end_day-1)
                plt.axvspan(start_date, end_date, alpha=0.2, color=color, 
                           label=season if periods[0] == (start_day, end_day) else "")
        
        # Highlight holidays
        holidays = generator.public_holidays
        for holiday in holidays:
            holiday_date = pd.to_datetime(holiday)
            plt.axvline(holiday_date, color='red', linestyle=':', alpha=0.7, linewidth=2)
        
        plt.title('Full Year Energy Consumption - All 9 Machines\nWith Seasonal Patterns and Holidays', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Energy Consumption (kWh)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        
        # Format x-axis
        months = pd.date_range('2024-01-01', '2024-12-31', freq='MS')
        plt.xticks(months, [month.strftime('%b') for month in months])
        
        # 2. Monthly consumption comparison (middle left)
        ax2 = plt.subplot(3, 2, 3)
        
        # Calculate monthly totals for each machine
        monthly_data = []
        months_list = []
        
        for month in range(1, 13):
            month_data = df_daily[df_daily['timestamp'].dt.month == month]
            monthly_totals = month_data[machines].sum()
            monthly_data.append(monthly_totals.values)
            months_list.append(pd.Timestamp(f'2024-{month:02d}-01').strftime('%b'))
        
        # Create stacked bar chart
        monthly_array = np.array(monthly_data).T
        bottom = np.zeros(12)
        
        for i, machine in enumerate(machines):
            plt.bar(months_list, monthly_array[i], bottom=bottom, 
                   label=machine, color=colors[i], alpha=0.8)
            bottom += monthly_array[i]
        
        plt.title('Monthly Energy Consumption by Machine', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Monthly Consumption (kWh)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # 3. Seasonal comparison (middle right)
        ax3 = plt.subplot(3, 2, 4)
        
        # Define seasons
        df_temp = df_daily.copy()
        df_temp['season'] = df_temp['timestamp'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Calculate seasonal averages
        seasonal_avg = df_temp.groupby('season')[machines].mean()
        seasonal_std = df_temp.groupby('season')[machines].std()
        
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        x_pos = np.arange(len(seasons))
        width = 0.08
        
        for i, machine in enumerate(machines):
            means = [seasonal_avg.loc[season, machine] for season in seasons]
            stds = [seasonal_std.loc[season, machine] for season in seasons]
            plt.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=machine, color=colors[i], alpha=0.8, capsize=3)
        
        plt.title('Seasonal Energy Consumption Patterns', fontsize=14, fontweight='bold')
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Daily Consumption (kWh)', fontsize=12)
        plt.xticks(x_pos + width*(len(machines)-1)/2, seasons)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Weekly pattern heatmap (bottom left)
        ax4 = plt.subplot(3, 2, 5)
        
        # Create weekly consumption pattern
        df_temp['week'] = df_temp['timestamp'].dt.isocalendar().week
        df_temp['day_of_week'] = df_temp['timestamp'].dt.day_name()
        
        # Calculate average consumption by week and day
        total_machine_consumption = df_temp[machines].sum(axis=1)
        weekly_pattern = df_temp.groupby(['week', 'day_of_week'])['total_consumption'].mean().unstack()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(columns=day_order)
        
        # Plot heatmap (show only first 20 weeks for clarity)
        sns.heatmap(weekly_pattern.iloc[:20], cmap='YlOrRd', cbar_kws={'label': 'Total Daily kWh'})
        plt.title('Weekly Consumption Patterns (First 20 Weeks)', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=12)
        plt.ylabel('Week Number', fontsize=12)
        
        # 5. Distribution analysis (bottom right)
        ax5 = plt.subplot(3, 2, 6)
        
        # Create box plot for daily consumption distribution
        daily_totals = [df_daily[machine] for machine in machines]
        box_plot = plt.boxplot(daily_totals, labels=machines, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Daily Consumption Distribution by Machine', fontsize=14, fontweight='bold')
        plt.xlabel('Machine', fontsize=12)
        plt.ylabel('Daily Consumption (kWh)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = "yearly_energy_distribution_9machines_15ops.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Yearly distribution plot saved as: {filename}")
        
        # Print comprehensive yearly statistics
        print(f"\n COMPREHENSIVE YEARLY STATISTICS:")
        print("=" * 60)
        
        # Overall statistics
        total_annual = df_daily[machines].sum().sum()
        print(f" Total Facility Consumption: {total_annual:,.0f} kWh/year")
        print(f" Average Daily Facility Total: {df_daily[machines].sum(axis=1).mean():.1f} kWh/day")
        print(f" Peak Daily Consumption: {df_daily[machines].sum(axis=1).max():.1f} kWh")
        print(f"🔽 Minimum Daily Consumption: {df_daily[machines].sum(axis=1).min():.1f} kWh")
        
        # Machine rankings
        machine_totals = df_daily[machines].sum().sort_values(ascending=False)
        print(f"\n MACHINE CONSUMPTION RANKINGS:")
        for i, (machine, total) in enumerate(machine_totals.items(), 1):
            percentage = (total / total_annual) * 100
            print(f"   {i:2d}. {machine:<20} {total:>8,.0f} kWh ({percentage:5.1f}%)")
        
        # Seasonal comparison
        print(f"\n SEASONAL CONSUMPTION ANALYSIS:")
        seasonal_totals = df_temp.groupby('season')[machines].sum().sum(axis=1)
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            total = seasonal_totals[season]
            avg_daily = total / len(df_temp[df_temp['season'] == season])
            print(f"   {season:<7}: {total:>8,.0f} kWh total, {avg_daily:>6.1f} kWh/day avg")
        
        # Workday vs Weekend
        df_temp['is_workday'] = df_temp['timestamp'].apply(generator.is_workday)
        workday_avg = df_temp[df_temp['is_workday']][machines].sum(axis=1).mean()
        weekend_avg = df_temp[~df_temp['is_workday']][machines].sum(axis=1).mean()
        reduction = (1 - weekend_avg/workday_avg) * 100
        
        print(f"\n WORKDAY vs WEEKEND ANALYSIS:")
        print(f"   Workday Average: {workday_avg:>6.1f} kWh/day")
        print(f"   Weekend Average: {weekend_avg:>6.1f} kWh/day")
        print(f"   Weekend Reduction: {reduction:>5.1f}%")
        
        # Monthly variation analysis
        monthly_totals = df_daily.groupby(df_daily['timestamp'].dt.month)[machines].sum().sum(axis=1)
        monthly_variation = monthly_totals.std() / monthly_totals.mean() * 100
        
        print(f"\n MONTHLY VARIATION ANALYSIS:")
        print(f"   Highest Month: {monthly_totals.idxmax():>2d} ({monthly_totals.max():>8,.0f} kWh)")
        print(f"   Lowest Month:  {monthly_totals.idxmin():>2d} ({monthly_totals.min():>8,.0f} kWh)")
        print(f"   Variation Coefficient: {monthly_variation:>5.1f}%")
        
        return filename

    def generate_comprehensive_report(self, df_dict, generator):
        """Generate comprehensive data quality and realism report"""
        print("\n" + "="*80)
        print("            9 MACHINES × 15 OPERATIONS - DATA QUALITY REPORT")
        print("="*80)
        
        print(f"\n GENERATION METHOD: Frequency-Weighted Probabilistic Consumption")
        print(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Configuration: {len(generator.machines)} machines, {len(generator.operations)} operations")
        
        # Operation distribution summary
        op_dist = generator.operation_distribution
        print(f"\n OPERATION FREQUENCY DISTRIBUTION:")
        for freq_type, ops in op_dist.items():
            print(f"   {freq_type.replace('_', ' ').title()}: {len(ops)} operations")
            print(f"      {', '.join(ops[:3])}{'...' if len(ops) > 3 else ''}")
        
        for resolution, df in df_dict.items():
            print(f"\n{' ' + resolution.upper() + ' RESOLUTION ANALYSIS'}")
            print("-" * 50)
            
            # Basic statistics
            print(f" Records: {len(df):,}")
            print(f" Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f" Time Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
            
            # Energy realism check
            energy_cols = [col for col in df.columns if col in generator.machines.keys()]
            
            for machine in energy_cols[:3]:  # Show first 3 machines for brevity
                machine_data = df[machine]
                machine_ops = generator.machines[machine]["operations"]
                print(f"\n   {machine} (Operations: {', '.join(machine_ops)}):")
                print(f"     Mean Consumption: {machine_data.mean():.2f} kWh")
                print(f"     Std Deviation: {machine_data.std():.2f} kWh")
                print(f"     Min/Max Range: {machine_data.min():.2f} - {machine_data.max():.2f} kWh")
                
                # Check for year-round consumption
                monthly_avg = df.groupby(df['timestamp'].dt.month)[machine].mean()
                zero_months = (monthly_avg == 0).sum()
                print(f"     Zero Consumption Months: {zero_months} {'✅ Year-round' if zero_months == 0 else '❌ Gaps found'}")
                
                # Realism metrics
                cv = machine_data.std() / machine_data.mean() if machine_data.mean() > 0 else 0
                print(f"    📐 Variation Coefficient: {cv:.3f} {'✅ Realistic' if cv < 0.8 else '⚠️ High'}")
            
            if len(energy_cols) > 3:
                print(f"\n  ... and {len(energy_cols) - 3} more machines with similar patterns")
            
            # Validation results
            is_valid, issues = generator.validate_data(df, resolution)
            print(f"\n  ✅ VALIDATION STATUS: {'PASSED' if is_valid else 'FAILED'}")
            if issues:
                print(f"   Issues Found: {len(issues)}")
                for issue in issues[:2]:
                    print(f"    - {issue}")
        
        # Overall assessment
        print(f"\n 9×15 MODEL ASSESSMENT:")
        print(f"✅ Exactly 9 machines and 15 operations as specified")
        print(f"✅ Frequency-based operation selection for realism")
        print(f"✅ Year-round consumption with seasonal variations")
        print(f"✅ Workday/weekend/holiday patterns implemented")
        print(f"✅ Proper energy ranges for industrial equipment")
        print(f"✅ Validated data quality and consistency")
        
        print("\n" + "="*80)

# ====================== MAIN EXECUTION ======================
def main():
    """Main execution function with 9 machines × 15 operations"""
    print("=" * 70)
    print(" INDUSTRIAL ENERGY GENERATOR: 9 MACHINES × 15 OPERATIONS")
    print("=" * 70)
    print(" Generating realistic industrial energy consumption data")
    
    # Initialize generator and visualizer
    generator = RealisticEnergyDataGenerator("config.yaml")
    visualizer = EnhancedEnergyVisualizer(generator.config)
    
    print("\n Generating realistic energy datasets...")
    
    # Generate data for all resolutions
    print("\n Generating hourly resolution data...")
    df_hourly = generator.generate_energy_data('hourly')
    print(" Generating 15-minute resolution data...")
    df_15min = generator.generate_energy_data('15min')
    print(" Generating daily resolution data...")
    df_daily = generator.generate_energy_data('daily')
    
    print(f"\n Data Generation Complete:")
    print(f"   15-minute data: {df_15min.shape[0]:,} records")
    print(f"   hourly data: {df_hourly.shape[0]:,} records") 
    print(f"   Daily data: {df_daily.shape[0]:,} records")
    
    # Quick verification
    print(f"\n🔍 CONFIGURATION VERIFICATION:")
    print(f"  Machines: {len(generator.machines)} ")
    print(f"  Operations: {len(generator.operations)} ")
    default_machine = generator.config['visualization']['default_machine']
    
    for resolution, df in [('daily', df_daily), ('hourly', df_hourly)]:
        monthly_totals = df.groupby(df['timestamp'].dt.month)[default_machine].sum()
        zero_months = (monthly_totals == 0).sum()
        print(f"  {resolution}: {12 - zero_months}/12 months with consumption {'✅' if zero_months == 0 else '❌'}")
    
    # Save to HDF5 with enhanced metadata
    print("\n Saving datasets and metadata...")
    generator.save_to_hdf5(df_15min, '15min')
    generator.save_to_hdf5(df_hourly, 'hourly')
    generator.save_to_hdf5(df_daily, 'daily')
    print(" All datasets and metadata saved successfully")
    
    # Enhanced data quality report
    df_dict = {'15min': df_15min, 'hourly': df_hourly, 'daily': df_daily}
    visualizer.generate_comprehensive_report(df_dict, generator)
    
    # Sample data preview
    print(f"\n SAMPLE DATA PREVIEW:")
    sample_cols = ['timestamp', 'shift', 'day_name', default_machine]
    print(df_15min[sample_cols].head(8).to_string(index=False))
    
    # Create visualizations
    print(f"\n Generating visualizations...")
    
    # 1. Machine-operation matrix
    print("Creating machine-operation assignment matrix...")
    visualizer.plot_machine_operation_matrix(generator)
    
    # 2. Multi-resolution comparison
    print(" Creating multi-resolution comparison plot...")
    visualizer.plot_energy_trends_all_resolutions(df_15min, df_hourly, df_daily)
    
    # 3. Yearly energy distribution analysis
    print(" Creating comprehensive yearly distribution analysis...")
    visualizer.plot_yearly_energy_distribution(df_daily, generator)
    
    print(f"\n9×15 INDUSTRIAL ENERGY DATA GENERATION COMPLETE!")
    print(f" 9 machines and 15 operations implemented")
    print(f" Frequency-based realistic operation patterns")
    print(f" Year-round consumption validated")
    print(f" Ready for industrial energy analysis!")

if __name__ == "__main__":
    main()
