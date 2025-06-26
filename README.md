# industrial-energy-consumption_15min_1h_1d_-resolution_germany_2024
High-resolution energy consumption data from 9 industrial machines across 15 operations in Germany (2024), with multi-temporal resolution (15-min, hourly, daily), shift annotations, public holiday adjustments and temporal patterns for manufacturing environments.

# Industrial Energy Data Generator

A Python framework for generating realistic industrial energy consumption datasets with 9 machines and 15 operations across multiple time resolutions.

## Overview

This tool generates synthetic energy consumption data for industrial manufacturing environments using probabilistic models that incorporate realistic operational patterns, shift schedules, seasonal variations, and equipment characteristics.

## Features

- **Exact Configuration**: 9 industrial machines, 15 operations
- **Multi-resolution Data**: 15-minute, hourly, and daily time series
- **Realistic Patterns**: Shift-based operations, seasonal adjustments, weekend/holiday reductions
- **Frequency Distribution**: High/medium/low frequency operation classification
- **Data Validation**: Comprehensive quality checks and consistency validation
- **Export Formats**: HDF5 data files with JSON metadata
- **Visualization**: Machine operation matrices, time series plots, yearly distribution analysis

## Requirements

```
numpy >= 1.20.0
pandas >= 1.3.0
h5py >= 3.1.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
PyYAML >= 5.4.0
scipy >= 1.7.0
```

## Installation

```bash
pip install numpy pandas h5py matplotlib seaborn PyYAML scipy
```

## Usage

### Basic Execution

```bash
python energy_generator.py
```

### File Structure

**Input Files:**
- `energy_generator.py` - Main generator script
- `config.yaml` - Configuration file (machines, operations, parameters)

**Output Files:**
- `energy_2024_9machines_15ops_daily.h5` - Daily resolution HDF5 data
- `energy_2024_9machines_15ops_hourly.h5` - Hourly resolution HDF5 data  
- `energy_2024_9machines_15ops_15min.h5` - 15-minute resolution HDF5 data
- `energy_2024_metadata_9m15o_daily.json` - Daily metadata
- `energy_2024_metadata_9m15o_hourly.json` - Hourly metadata
- `energy_2024_metadata_9m15o_15min.json` - 15-minute metadata
- `machine_operation_matrix_9x15.png` - Operation assignment visualization
- `yearly_energy_distribution_9machines_15ops.png` - Yearly consumption analysis
- `CNC_Mill_1_9machines_15ops_7days.png` - Sample machine time series

## Configuration

The `config.yaml` file defines:

### Machines (9 total)
```yaml
machines:
  CNC_Mill_1:
    type: "CNC"
    base_kWh: 1.2
    operations: ["Milling", "Drilling"]
  Injection_Molder_2:
    type: "Molder"
    base_kWh: 2.5
    operations: ["Injection_Molding"]
  # ... 7 more machines
```

### Operations (15 total)
```yaml
operations:
  Milling:
    kWh_range: [8, 15]
    noise: 0.1
  Drilling:
    kWh_range: [5, 10]
    noise: 0.15
  # ... 13 more operations
```

### Operation Frequency Distribution
- **High Frequency**: Milling, Assembly, Packaging, Testing, Quality_Control
- **Medium Frequency**: Drilling, Cutting, Injection_Molding, Cooling, Engraving
- **Low Frequency**: Welding, Heat_Treatment, Drying, Painting, Stamping

### Shift Configuration
- **Morning**: 6AM-2PM (80% operation probability)
- **Afternoon**: 2PM-10PM (90% operation probability)
- **Night**: 10PM-6AM (30% operation probability)

### Seasonal Adjustments
- **Winter**: 20% higher base load, 10% lower operations
- **Summer**: 10% higher base load, normal operations
- **Spring/Fall**: Normal base load and operations

## Data Format

### HDF5 Structure
```
energy_2024_9machines_15ops_daily.h5
├── timestamps              # Unix epoch timestamps
├── time_info/
│   ├── shift              # Shift assignments
│   ├── shift_label        # Shift descriptions
│   └── day_name           # Day of week
├── machines/
│   ├── CNC_Mill_1         # Machine energy data
│   ├── Injection_Molder_2
│   └── ... (7 more)
└── metadata/
    ├── operations/        # Operation specifications
    ├── shifts             # Shift configurations
    └── holidays           # Public holidays
```

### Data Columns
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64 | Record timestamp |
| shift | string | Current shift (Morning/Afternoon/Night) |
| day_name | string | Day of week |
| {machine_name} | float64 | Energy consumption (kWh) |

## Validation

The generator performs comprehensive validation:

- **Configuration Validation**: Ensures exactly 9 machines and 15 operations
- **Data Quality Checks**: Validates year-round consumption, realistic ranges
- **Pattern Verification**: Confirms weekend reductions, seasonal variations
- **Statistical Analysis**: Checks variation coefficients, outlier detection

## Customization

### Adding Machines
Modify the `machines` section in `config.yaml`:
```yaml
machines:
  New_Machine_10:
    type: "Custom"
    base_kWh: 2.0
    operations: ["Custom_Operation"]
```

### Modifying Operations
Update the `operations` section:
```yaml
operations:
  Custom_Operation:
    kWh_range: [5, 12]
    noise: 0.2
```

### Adjusting Holidays
Update the `public_holidays` list:
```yaml
public_holidays:
  - "2024-01-01"  # New Year's Day
  - "2024-12-25"  # Christmas Day
```

## Generated Visualizations

1. **Machine Operation Matrix**: Shows which operations are assigned to which machines with frequency indicators
2. **Multi-Resolution Time Series**: Displays 15-minute, hourly, and daily patterns for sample periods
3. **Yearly Distribution Analysis**: Comprehensive 6-panel analysis including:
   - Full year consumption trends with seasonal backgrounds
   - Monthly consumption by machine (stacked bars)
   - Seasonal consumption patterns (grouped bars)
   - Weekly consumption heatmap
   - Daily consumption distribution (box plots)

## Data Statistics

The generator provides detailed statistics:
- Total facility consumption (kWh/year)
- Machine consumption rankings
- Seasonal consumption analysis
- Workday vs weekend comparisons
- Monthly variation analysis

## Troubleshooting

### Common Issues

**Import Errors**: Ensure all required packages are installed
```bash
pip install --upgrade numpy pandas h5py matplotlib seaborn PyYAML scipy
```

**Configuration Errors**: Verify `config.yaml` syntax and machine/operation counts

**Memory Issues**: For large datasets, consider generating one resolution at a time

### Validation Failures

The system validates:
- Negative energy values (should be zero)
- Energy ranges (0-1000 kWh limits)
- Year-round consumption (no zero months)
- Realistic variation coefficients (<2.0)

## Performance

**Generation Time**: 
- 15-minute resolution: ~2-3 minutes (35,040 records)
- Hourly resolution: ~30 seconds (8,760 records)
- Daily resolution: ~5 seconds (365 records)

**File Sizes**:
- 15-minute HDF5: ~15 MB
- Hourly HDF5: ~4 MB
- Daily HDF5: ~150 KB

## License

This project is provided as-is for industrial energy analysis and research purposes.

## Support

For issues or questions:
1. Verify configuration syntax in `config.yaml`
2. Check console output for validation warnings
3. Review generated metadata files for data quality reports
4. Examine visualization outputs for pattern verification
- **Documentation**: See docs/ directory for detailed guides
- **Issues**: Use GitHub Issues for bug reports
- **Questions**: Contact: dalim@rptu.de

## Version History

- **v1.2.0**: 9 machines × 15 operations implementation
- **v1.1.0**: Multi-resolution support and enhanced validation
- **v1.0.0**: Initial release
