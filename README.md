# Apparel Production Scheduler

This project implements a forward planning scheduler for a simplified apparel production process with three sequential stages: **Cut → Sew → Pack**. The system uses Monte Carlo simulation with intelligent heuristics to find an optimal schedule that minimizes order lateness.

## Features

- **Order Scheduling**: Automatically schedules orders across multiple machines for cutting, sewing, and packing.
- **Monte Carlo Simulation**: Optimizes the schedule by running multiple simulations with randomized order sequences.
- **Visualization**: Generates Gantt charts, lateness charts, and machine utilization charts for the production schedule.
- **Metrics Evaluation**: Provides detailed metrics such as total lateness, average lateness, and on-time completion rate.
- **Conflict Analysis**: Detects overlaps and sequential violations in the schedule.

## Scheduling Strategies

The scheduler employs six sophisticated strategies in its Monte Carlo simulation:

1. **Product Type Batching**: Groups orders by product type and sub-sorts by deadline to minimize setup times.
2. **Out-of-Factory Priority**: Prioritizes orders without post-cutting delays while considering deadlines.
3. **Modified Critical Ratio**: Uses deadline/processing time ratio with adjustment for out-of-factory delay requirements.
4. **Balanced Operation Load**: Optimizes machine utilization by balancing workload across stages.
5. **Weighted Combination**: Uses a multi-factor scoring system considering deadlines, processing times, delays, and setup benefits.
6. **Two-Phase Dynamic Slack**: Separates and prioritizes critical orders based on slack time, with product type optimization for non-critical orders.

## Performance Metrics

The scheduler evaluates schedules based on several metrics:
- Total orders completed on time
- Average lateness across all orders
- Maximum lateness for any order
- Machine utilization rates
- Product type-specific performance
- Setup time optimization

## Installation

1. Clone the repository or download the project files.
2. Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Scheduler
Run the main script with the following command:
```bash
python main.py --file Data.csv --simulations 500 --output results
```

### Command-Line Arguments
- `--file`: Path to the CSV file containing order data (default: `Data.csv`).
- `--simulations`: Number of Monte Carlo simulations to run (default: `500`).
- `--output`: Directory to save the results (default: `results`).
- `--test`: Run unit tests before executing the scheduler.

### Outputs
- **`order_schedule.csv`**: Detailed schedule for each order.
- **`best_sequence.txt`**: Best order sequence found during simulation.
- **`summary.txt`**: Summary of lateness metrics.
- **`schedule_gantt.png`**: Gantt chart visualization.
- **`schedule_lateness.png`**: Lateness chart visualization.
- **`machine_utilization.png`**: Machine utilization rates visualization.

### Analyzing Overlaps and Violations
After running the scheduler, you can analyze the schedule for overlaps and sequential violations:
```bash
python results/overlap.py
```
This will display any conflicts or violations in the schedule.

## Results Analysis

The output provides comprehensive analytics including:
- Detailed Gantt charts showing machine utilization
- Lateness distribution across orders
- Machine utilization rates
- Product type performance analysis
- Setup time impact assessment

## Testing

Run the tests using `pytest`:
```bash
pytest test/unit_tests.py
```

## Project Structure

```
Manny/
├── task/
│   ├── apparel_scheduler.py   # Core scheduling logic
│   ├── main.py                # Main script to run the scheduler
│   ├── Data.csv               # Sample input data
│   ├── results/               # Output directory for results
│   ├── test/                  # Unit tests for the scheduler
│   │   ├── unit_tests.py      # Unit test cases
│   │   └── __init__.py        # Makes the test directory a package
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- pytest

## License

This project is licensed under the MIT License.
