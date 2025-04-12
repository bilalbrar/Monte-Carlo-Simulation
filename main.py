"""
Apparel Production Scheduler - Main Script
"""
import pandas as pd
import numpy as np
import random
import time
import os
import argparse
import matplotlib.pyplot as plt
from apparel_scheduler import Scheduler
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task.test.unit_tests import test_order_class, test_machine_class, test_factory_simple_schedule, test_factory_with_delay

def run_tests():
    """Run all test cases."""
    test_order_class()
    test_machine_class()
    test_factory_simple_schedule()
    test_factory_with_delay()
    print("All tests passed successfully!")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Apparel Production Scheduler')
    parser.add_argument('--file', type=str, default='Data.csv',
                        help='Path to the CSV file with order data')
    parser.add_argument('--simulations', type=int, default=500,
                        help='Number of Monte Carlo simulations to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--test', action='store_true',
                        help='Run tests before execution')
    return parser.parse_args()

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_results(scheduler, output_dir):
    """Save detailed results to CSV files."""
    # Get detailed schedule
    detailed = scheduler.get_detailed_schedule()
    
    # Create DataFrame with order details
    order_df = pd.DataFrame(detailed['order_details'])
    order_df.to_csv(f"{output_dir}/order_schedule.csv", index=False)
    
    # Create sequence file
    with open(f"{output_dir}/best_sequence.txt", 'w') as f:
        f.write(','.join(detailed['best_sequence']))
    
    # Create summary file
    metrics = scheduler.best_metrics
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"Total orders: {metrics['total_orders']}\n")
        f.write(f"Orders completed on time: {metrics['on_time_orders']} ({metrics['on_time_orders']/metrics['total_orders']*100:.1f}%)\n")
        f.write(f"Average lateness: {metrics['avg_lateness']:.2f} time units\n")
        f.write(f"Maximum lateness: {metrics['max_lateness']} time units\n")
        f.write(f"Total lateness: {metrics['total_lateness']} time units\n")
        
        f.write("\nLateness by Order (only late orders):\n")
        late_orders = [(order_id, lateness) for order_id, lateness in metrics["order_lateness"].items() if lateness > 0]
        late_orders.sort(key=lambda x: x[1], reverse=True)
        
        for order_id, lateness in late_orders:
            f.write(f"{order_id}: {lateness} time units late\n")
        
        f.write("\nLateness Analysis by Product Type:\n")
        order_details = detailed['order_details']
        df = pd.DataFrame(order_details)
        type_analysis = df.groupby('product_type').agg({
            'lateness': ['mean', 'max', 'count']
        }).round(2)
        f.write("\nProduct Type Analysis:\n")
        f.write(type_analysis.to_string())

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Run tests if requested
    if args.test:
        run_tests()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    
    # Print start message
    print(f"=== Apparel Production Scheduler ===")
    print(f"Loading data from: {args.file}")
    print(f"Running {args.simulations} Monte Carlo simulations")
    print(f"Results will be saved to: {output_dir}")
    print("="*36)
    
    # Start timer
    start_time = time.time()
    
    # Load order data from CSV
    try:
        orders_df = pd.read_csv(args.file)
        print(f"Loaded {len(orders_df)} orders from the CSV file")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Create scheduler
    scheduler = Scheduler(orders_df)
    
    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation with {args.simulations} iterations...")
    best_metrics = scheduler.run_monte_carlo_simulation(num_simulations=args.simulations)
    
    # Print results
    print("\nBest Schedule Results:")
    print(f"Total orders: {best_metrics['total_orders']}")
    print(f"Orders completed on time: {best_metrics['on_time_orders']} ({best_metrics['on_time_orders']/best_metrics['total_orders']*100:.1f}%)")
    print(f"Average lateness: {best_metrics['avg_lateness']:.2f} time units")
    print(f"Maximum lateness: {best_metrics['max_lateness']} time units")
    
    # Save results
    print("\nSaving results...")
    save_results(scheduler, output_dir)
    
    # Create visualisations
    print("Generating visualisations...")
    scheduler.visualise_schedule(
        save_gantt=f"{output_dir}/schedule_gantt.png",
        save_lateness=f"{output_dir}/schedule_lateness.png"
    )
    # Add machine utilisation visualisation
    scheduler.create_machine_utilisation_chart(
        save_path=f"{output_dir}/machine_utilisation.png"
    )
    
    # Add machine utilisation visualisation
    scheduler.create_machine_utilisation_chart(
        save_path=f"{output_dir}/machine_utilisation.png"
    )
    
    # End timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    print(f"\nAll results saved to the '{output_dir}' directory")


if __name__ == "__main__":
    main()