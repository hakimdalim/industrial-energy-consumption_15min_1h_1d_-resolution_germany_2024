# industrial-energy-consumption_15min_1h_1d_-resolution_germany_2024
High-resolution energy consumption data from 9 industrial machines across 15 operations in Germany (2024), with multi-temporal resolution (15-min, hourly, daily), shift annotations, public holiday adjustments and temporal patterns for manufacturing environments.


## Overview

This tool generates synthetic energy consumption data for industrial facilities with 9 machines and 15 operations, incorporating realistic operational patterns, shift schedules, seasonal variations, and holiday effects. The generated datasets support multiple time resolutions and include comprehensive validation and analysis capabilities.

## Features

- **Machine Configuration**: 9 industrial machines (CNC mills, injection molders, laser cutters, robots, ovens, packaging lines, welding cells, painting booths, hydraulic presses)
- **Operation Management**: 15 distinct operations with frequency-based selection (high/medium/low frequency distribution)
- **Time Resolutions**: 15-minute, hourly, and daily data aggregation
- **Realistic Patterns**: Shift-based operations, seasonal adjustments, weekend/holiday reductions
- **Data Export**: HDF5 format with embedded metadata and JSON metadata files
- **Validation Framework**: Comprehensive data quality checks and realism verification
- **Visualization Suite**: Multi-resolution plots, yearly distribution analysis, machine-operation matrices

## Requirements

```
Python >= 3.8
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
git clone https://github.com/hakimdalim/industrial-energy-consumption_15min_1h_1d_-resolution_germany_2024
.git
cd industrial-energy-consumption_15min_1h_1d_-resolution_germany_2024
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from energy_generator import RealisticEnergyDataGenerator

# Initialize generator
generator = RealisticEnergyDataGenerator("config.yaml")

# Generate daily resolution data
df_daily = generator.generate_energy_data('daily')

# Save to HDF5 with metadata
generator.save_to_hdf5(df_daily, 'daily')
```

### Command Line Execution

```bash
python energy_generator.py
```

### Data Analysis

```python
from h5_analyzer import analyze_h5_file_jupyter

# Analyze generated data
analyzer = analyze_h5_file_jupyter("energy_2024_9machines_15ops_daily.h5")
```

## Configuration

The system uses YAML configuration files to define machine specifications, operation parameters, and generation settings.

### Machine Configuration

```yaml
machines:
  CNC_Mill_1:
    type: "CNC"
    base_kWh: 1.2
    operations: ["Milling", "Drilling"]
  
  Injection_Molder_2:
    type: "Molder"
    base_kWh: 2.5
    operations: ["Injection_Molding", "Cooling"]
```

### Operation Parameters

```yaml
operations:
  Milling:
    kWh_range: [8, 15]
    noise: 0.1
  
  Injection_Molding:
    kWh_range: [10, 20]
    noise: 0.25
```

### Operation Frequency Distribution

```yaml
operation_distribution:
  high_frequency: ["Milling", "Assembly", "Packaging", "Testing", "Quality_Control"]
  medium_frequency: ["Drilling", "Cutting", "Injection_Molding", "Cooling", "Engraving"]
  low_frequency: ["Welding", "Heat_Treatment", "Drying", "Painting", "Stamping"]
```

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
│   ├── CNC_Mill_1         # Energy consumption data
│   ├── Injection_Molder_2
│   └── ...
└── metadata/
    ├── shifts             # Shift configurations
    ├── holidays           # Public holidays
    ├── operations/        # Operation specifications
    └── operation_distribution/
```

### DataFrame Structure

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime64 | Record timestamp |
| shift | object | Current shift (Morning/Afternoon/Night) |
| shift_label | object | Descriptive shift label |
| day_name | object | Day of week |
| {machine_name} | float64 | Energy consumption (kWh) |

## Validation and Quality Assurance

The framework includes comprehensive validation:

- **Configuration Validation**: Ensures exactly 9 machines and 15 operations
- **Data Quality Checks**: Negative values, range validation, year-round consumption
- **Realism Verification**: Weekend patterns, seasonal variations, statistical consistency
- **Temporal Validation**: Missing timestamps, shift consistency

### Realism Metrics

Each machine receives a realism score (0-100) based on:
- Year-round consumption (25 points)
- Weekend reduction patterns (25 points)
- Reasonable statistical variation (25 points)
- Non-negative values (25 points)

## Output Files

### Generated Files

- `energy_2024_9machines_15ops_{resolution}.h5` - HDF5 data files
- `energy_2024_metadata_9m15o_{resolution}.json` - Comprehensive metadata
- `machine_operation_matrix_9x15.png` - Assignment visualization
- `{machine}_9machines_15ops_{days}days.png` - Multi-resolution plots
- `yearly_energy_distribution_9machines_15ops.png` - Annual analysis

### Metadata Content

- Dataset information and generation parameters
- Machine and operation specifications
- Statistical summaries and quality metrics
- Validation results and realism assessments
- Temporal pattern analysis

## API Reference

### RealisticEnergyDataGenerator

**Methods:**
- `generate_energy_data(resolution)`: Generate data at specified resolution
- `save_to_hdf5(df, resolution)`: Export data to HDF5 format
- `validate_data(df, resolution)`: Perform quality validation
- `generate_metadata(df, resolution)`: Create comprehensive metadata

### EnhancedEnergyVisualizer

**Methods:**
- `plot_machine_operation_matrix(generator)`: Machine-operation assignments
- `plot_energy_trends_all_resolutions()`: Multi-resolution comparison
- `plot_yearly_energy_distribution()`: Annual consumption analysis
- `generate_comprehensive_report()`: Data quality assessment

## Testing

```bash
# Run validation tests
python -m pytest tests/

# Verify installation
python -c "from energy_generator import RealisticEnergyDataGenerator; print('Installation verified')"
```

## Customization

### Adding Machines

```yaml
machines:
  Custom_Machine_10:
    type: "Custom"
    base_kWh: 2.8
    operations: ["Custom_Operation"]
```

### Defining Operations

```yaml
operations:
  Custom_Operation:
    kWh_range: [5, 12]
    noise: 0.15
```

### Holiday Configuration

```yaml
public_holidays:
  - "2024-01-01"  # New Year's Day
  - "2024-12-25"  # Christmas Day
```

## Performance Characteristics

- **15-minute resolution**: ~35,000 records/year
- **Hourly resolution**: ~8,760 records/year
- **Daily resolution**: 365 records/year
- **Generation time**: <30 seconds for full year
- **Memory usage**: <100MB for all resolutions
- **File sizes**: 5-50MB depending on resolution

## Troubleshooting

### Common Issues

1. **Configuration validation errors**: Verify exactly 9 machines and 15 operations in config.yaml
2. **Missing dependencies**: Install all required packages via requirements.txt
3. **File permission errors**: Ensure write permissions in output directory
4. **Memory issues**: Reduce data generation scope or increase available RAM

### Validation Failures

Check console output for specific validation issues:
- Negative values indicate configuration problems
- Zero consumption months suggest unrealistic parameters
- High variation coefficients indicate excessive noise

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Create Pull Request

## Support

- **Documentation**: See docs/ directory for detailed guides
- **Issues**: Use GitHub Issues for bug reports
- **Questions**: Contact: dalim@rptu.de

## Version History

- **v1.2.0**: 9 machines × 15 operations implementation
- **v1.1.0**: Multi-resolution support and enhanced validation
- **v1.0.0**: Initial release
