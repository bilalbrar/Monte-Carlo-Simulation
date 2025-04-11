"""
Apparel Production Scheduler

This module implements a forward planning scheduler for a simplified apparel production
process with three sequential stages: Cut → Sew → Pack.

The system uses Monte Carlo simulation with intelligent heuristics to find an optimal schedule 
that minimizes order lateness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
import random
import time
import copy
from typing import List, Dict, Tuple


class Order:
    """
    Represents an apparel production order with all necessary attributes.
    """
    def __init__(self, order_id: str, product_type: str, cut_time: int, 
                 sew_time: int, pack_time: int, deadline: int, 
                 requires_delay: bool):
        self.order_id = order_id
        self.product_type = product_type
        self.cut_time = cut_time
        self.sew_time = sew_time
        self.pack_time = pack_time
        self.deadline = deadline
        self.requires_delay = requires_delay
        
        # Scheduling information (to be filled during scheduling)
        self.cut_start_time = None
        self.cut_end_time = None
        self.cut_machine_id = None
        
        self.sew_start_time = None
        self.sew_end_time = None
        self.sew_machine_id = None
        
        self.pack_start_time = None
        self.pack_end_time = None
        
        self.completion_time = None
        self.lateness = None
        
        # Calculate total processing time for prioritization
        self.total_processing_time = cut_time + sew_time + pack_time
    
    def is_scheduled(self) -> bool:
        """Check if the order has been fully scheduled."""
        return (self.cut_start_time is not None and 
                self.sew_start_time is not None and 
                self.pack_start_time is not None)
    
    def calculate_lateness(self) -> int:
        """Calculate how late the order is (in time units)."""
        if self.completion_time is None:
            raise ValueError("Order not fully scheduled yet")
        self.lateness = max(0, self.completion_time - self.deadline)
        return self.lateness
    
    def __repr__(self) -> str:
        return f"Order({self.order_id}, type={self.product_type}, deadline={self.deadline})"


class Machine:
    """
    Represents a production machine at a specific stage.
    """
    def __init__(self, machine_id: str, stage: str):
        self.machine_id = machine_id
        self.stage = stage
        self.schedule = []  # List of (start_time, end_time, order_id) tuples
        self.last_product_type = None  # For tracking product type changes
        self.next_available_time = 0  # Track when machine becomes available
    
    def is_available(self, start_time: int, duration: int) -> Tuple[bool, int]:
        """
        Check if the machine is available at the given time for the specified duration.
        Returns (is_available, next_available_time).
        """
        if not self.schedule:
            return True, max(self.next_available_time, start_time)

        sorted_schedule = sorted(self.schedule, key=lambda x: x[0])
        proposed_end = start_time + duration

        # Check if we can schedule before the first task
        if sorted_schedule[0][0] >= proposed_end:
            return True, start_time

        # Look for gaps between tasks
        for i in range(len(sorted_schedule) - 1):
            curr_end = sorted_schedule[i][1]
            next_start = sorted_schedule[i + 1][0]
            
            possible_start = max(curr_end, start_time)
            if possible_start + duration <= next_start:
                return True, possible_start

        # Check if we can schedule after the last task
        last_end = sorted_schedule[-1][1]
        possible_start = max(last_end, start_time)
        if possible_start + duration <= proposed_end:
            return True, possible_start

        return False, last_end
    
    def add_task(self, start_time: int, end_time: int, order_id: str, product_type: str = None) -> None:
        """
        Schedule a task on this machine.
        """
        # Ensure no overlaps
        for task_start, task_end, _ in self.schedule:
            if not (end_time <= task_start or start_time >= task_end):
                raise ValueError(f"Task overlaps with an existing task on {self.machine_id}: "
                                 f"New task ({start_time}-{end_time}) overlaps with ({task_start}-{task_end})")

        # Add task to schedule
        self.schedule.append((start_time, end_time, order_id))
        self.schedule.sort(key=lambda x: x[0])  # Keep sorted by start time
        
        # Update machine state
        self.next_available_time = max(self.next_available_time, end_time)
        if self.stage == "cut" and product_type is not None:
            self.last_product_type = product_type
    
    def get_setup_time(self, product_type: str) -> int:
        """
        Return the setup time needed if switching to a new product type.
        Only applies to cutting machines.
        """
        if self.stage != "cut":
            return 0
        
        if self.last_product_type is None or self.last_product_type == product_type:
            return 0
        else:
            # 10 time units for product type switch on cutting machines
            return 10
    
    def __repr__(self) -> str:
        return f"Machine({self.machine_id}, {self.stage}, tasks={len(self.schedule)})"


class Factory:

    def __init__(self):
        # Initialize machines (tasks cannot be split across machines)
        self.cutting_machines = [Machine(f"Cut_{i+1}", "cut") for i in range(2)]  # 2 cutting tables
        self.sewing_machines = [Machine(f"Sew_{i+1}", "sew") for i in range(3)]  # 3 sewing stations
        self.packing_machines = [Machine(f"Pack_{i+1}", "pack") for i in range(1)]  # 1 packing station
        
        self.all_machines = self.cutting_machines + self.sewing_machines + self.packing_machines
        
        # Keep track of orders
        self.orders = {}
        
        # Factory parameters
        self.post_cutting_delay = 48  # Time units delay after cutting for some orders

    def add_order(self, order: Order) -> None:
        """Add an order to the factory."""
        self.orders[order.order_id] = order

    def _calculate_gap_utilization(self, machine: Machine, start_time: int, duration: int) -> float:
        """
        Calculate how well a task utilizes existing gaps in the machine schedule.
        Returns a score from 0 to 1, where 1 means perfect utilization.
        """
        if not machine.schedule:
            return 0  # No gaps to utilize in an empty schedule
        
        end_time = start_time + duration
        
        # Sort the schedule
        sorted_schedule = sorted(machine.schedule, key=lambda x: x[0])
        
        # Check if this task fits perfectly between two existing tasks
        for i in range(len(sorted_schedule) - 1):
            curr_end = sorted_schedule[i][1]
            next_start = sorted_schedule[i + 1][0]
            
            # If we're starting exactly after a task and ending exactly before another
            if start_time == curr_end and end_time == next_start:
                return 1.0  # Perfect fit
            
            # If we're starting after a task ends and ending before another starts
            if start_time >= curr_end and end_time <= next_start:
                gap_size = next_start - curr_end
                fill_ratio = duration / gap_size
                return fill_ratio  # How much of the gap we're filling
        
        # Check if we're filling a gap at the beginning
        if sorted_schedule[0][0] > 0 and end_time <= sorted_schedule[0][0]:
            gap_size = sorted_schedule[0][0]
            fill_ratio = duration / gap_size
            return fill_ratio
        
        # Check if we're filling a gap at the end
        if start_time >= sorted_schedule[-1][1]:
            return 0.5  # Medium priority for appending at the end
        
        return 0  # Not utilizing any gap
    
    def _schedule_cutting_stage(self, order: Order) -> Tuple[int, Machine, int]:
        """Schedule the cutting stage for an order with proper setup time handling."""
        best_start = float('inf')
        best_machine = None
        best_setup_time = 0

        # First try machines already set up for this product type
        for machine in self.cutting_machines:
            if machine.last_product_type == order.product_type:
                _, next_time = machine.is_available(0, order.cut_time)
                if next_time < best_start:
                    best_start = next_time
                    best_machine = machine
                    best_setup_time = 0
        
        # If no matching machine found, try all machines
        if best_machine is None:
            for machine in self.cutting_machines:
                setup_time = machine.get_setup_time(order.product_type)
                _, next_time = machine.is_available(0, order.cut_time + setup_time)
                if next_time < best_start:
                    best_start = next_time
                    best_machine = machine
                    best_setup_time = setup_time

        if best_machine is None:
            raise ValueError(f"No cutting machine available for order {order.order_id}")

        return best_start, best_machine, best_setup_time

    def _schedule_sewing_stage(self, order: Order, earliest_start: int) -> Tuple[int, Machine]:
        """
        Schedule the sewing stage for an order, ensuring it starts as soon as cutting is finished
        (or after the post-cutting delay if applicable).
        """
        best_start = float('inf')
        best_machine = None

        for machine in self.sewing_machines:
            # Ensure sewing starts no earlier than the earliest_start
            _, next_time = machine.is_available(earliest_start, order.sew_time)
            adjusted_start = max(next_time, earliest_start)

            # Prioritize the earliest adjusted start time
            if adjusted_start < best_start:
                best_start = adjusted_start
                best_machine = machine

        if best_machine is None:
            raise ValueError(f"No sewing machine available for order {order.order_id}")

        return best_start, best_machine

    def _schedule_packing_stage(self, order: Order, earliest_start: int) -> Tuple[int, Machine]:
        """Schedule the packing stage for an order."""
        machine = self.packing_machines[0]
        _, next_time = machine.is_available(earliest_start, order.pack_time)
        return next_time, machine
    
    def schedule_order(self, order: Order) -> None:
        """Schedule an order through all production stages."""
        try:
            # Step 1: Schedule cutting stage
            cut_start, cut_machine, setup_time = self._schedule_cutting_stage(order)
            
            # Calculate actual cutting start time after setup
            actual_cut_start = cut_start + setup_time  # Add setup time if needed
            actual_cut_end = actual_cut_start + order.cut_time
            
            # Update order with cutting information
            order.cut_start_time = actual_cut_start
            order.cut_end_time = actual_cut_end
            order.cut_machine_id = cut_machine.machine_id
            
            # Add task to cutting machine
            cut_machine.add_task(cut_start, actual_cut_end, order.order_id, order.product_type)

            # Step 2: Schedule sewing stage
            earliest_sew_start = order.cut_end_time
            if order.requires_delay:
                earliest_sew_start += self.post_cutting_delay

            sew_start, sew_machine = self._schedule_sewing_stage(order, earliest_sew_start)
            
            # Update order with sewing information
            order.sew_start_time = sew_start
            order.sew_end_time = sew_start + order.sew_time
            order.sew_machine_id = sew_machine.machine_id
            sew_machine.add_task(sew_start, order.sew_end_time, order.order_id)

            # Step 3: Schedule packing stage
            pack_start, pack_machine = self._schedule_packing_stage(order, order.sew_end_time)

            # Update order with packing information
            order.pack_start_time = pack_start
            order.pack_end_time = pack_start + order.pack_time
            order.completion_time = order.pack_end_time
            pack_machine.add_task(pack_start, order.pack_end_time, order.order_id)

            # Calculate lateness
            order.calculate_lateness()

        except Exception as e:
            self._reset_order_schedule(order)
            raise ValueError(f"Failed to schedule order {order.order_id}: {str(e)}")

    def schedule_all_orders(self, order_sequence: List[str]) -> None:
        """Schedule all orders in the specified sequence."""
        # Reset all machine schedules
        for machine in self.all_machines:
            machine.schedule = []
            machine.last_product_type = None
            machine.next_available_time = 0
        
        # Process orders in sequence
        for order_id in order_sequence:
            if order_id in self.orders:
                try:
                    self.schedule_order(self.orders[order_id])
                except ValueError as e:
                    print(f"Warning: Could not schedule order {order_id}: {str(e)}")

    def evaluate_schedule(self) -> Dict:
        """Evaluate the current schedule and return metrics."""
        if not all(order.is_scheduled() for order in self.orders.values()):
            raise ValueError("Not all orders are scheduled")
        
        # Calculate metrics
        total_orders = len(self.orders)
        on_time_orders = sum(1 for order in self.orders.values() if order.lateness == 0)
        total_lateness = sum(order.lateness for order in self.orders.values())
        avg_lateness = total_lateness / total_orders if total_orders > 0 else 0
        max_lateness = max(order.lateness for order in self.orders.values()) if total_orders > 0 else 0
        
        # Detailed order lateness
        order_lateness = {order.order_id: order.lateness for order in self.orders.values()}
        
        return {
            "total_orders": total_orders,
            "on_time_orders": on_time_orders,
            "total_lateness": total_lateness,
            "avg_lateness": avg_lateness,
            "max_lateness": max_lateness,
            "order_lateness": order_lateness
        }

    def _reset_order_schedule(self, order: Order) -> None:
        """Reset all scheduling information for an order."""
        order.cut_start_time = None
        order.cut_end_time = None
        order.cut_machine_id = None
        order.sew_start_time = None
        order.sew_end_time = None
        order.sew_machine_id = None
        order.pack_start_time = None
        order.pack_end_time = None
        order.completion_time = None
        order.lateness = None

        # Also remove this order from all machine schedules
        for machine in self.all_machines:
            machine.schedule = [(s, e, o) for s, e, o in machine.schedule if o != order.order_id]
            if machine.stage == 'cut' and machine.last_product_type == order.product_type:
                # Don't reset product type when removing a single order
                pass


class Scheduler:
    """
    Main scheduler class that handles Monte Carlo simulations and optimization.
    """
    def __init__(self, orders_data: pd.DataFrame):
        self.orders_data = orders_data
        self.factory = Factory()
        
        # Load orders into factory
        for _, row in orders_data.iterrows():
            order = Order(
                order_id=row['order_id'],
                product_type=row['Product type'],
                cut_time=row['cut time'],
                sew_time=row['sew time'],
                pack_time=row['pack time'],
                deadline=row['deadline'],
                requires_delay=row['requires_out_of_factory_delay']
            )
            self.factory.add_order(order)
        
        # Keep track of the best schedule found
        self.best_schedule = None
        self.best_metrics = None
    
    def run_monte_carlo_simulation(self, num_simulations: int = 100) -> Dict:
        """
        Run Monte Carlo simulations with various heuristics to find the best schedule.
        """
        order_ids = list(self.factory.orders.keys())
        
        start_time = time.time()
        valid_schedule_found = False  # Track if a valid schedule is found
        
        for i in range(num_simulations):
            # Replace the random strategy section with these approaches
            if i % 4 == 0:
                # Product type batching with deadline sub-sorting
                sequence = sorted(order_ids, 
                                  key=lambda x: (self.factory.orders[x].product_type, 
                                                 self.factory.orders[x].deadline))
            elif i % 4 == 1:
                # Out-of-factory priority with deadline sorting
                sequence = sorted(order_ids, 
                                  key=lambda x: (not self.factory.orders[x].requires_delay,
                                                 self.factory.orders[x].deadline))
            elif i % 4 == 2:
                # Modified critical ratio with out-of-factory consideration
                sequence = sorted(order_ids, 
                                  key=lambda x: (self.factory.orders[x].deadline / 
                                                 (self.factory.orders[x].total_processing_time * 
                                                  (1.5 if self.factory.orders[x].requires_delay else 1))))
            elif i % 4 == 3:
                # Balanced operation load
                sequence = sorted(order_ids, 
                                  key=lambda x: max(self.factory.orders[x].cut_time,
                                                    self.factory.orders[x].sew_time,
                                                    self.factory.orders[x].pack_time) / 
                                                self.factory.orders[x].total_processing_time)
            
            try:
                self.factory.schedule_all_orders(sequence)
                metrics = self.factory.evaluate_schedule()
                valid_schedule_found = True  # Mark that a valid schedule was found
                
                if self.best_metrics is None or metrics["avg_lateness"] < self.best_metrics["avg_lateness"]:
                    self.best_metrics = metrics
                    self.best_schedule = sequence.copy()
                    print(f"New best schedule found with avg lateness: {metrics['avg_lateness']:.2f} (strategy: {i%4})")
            except Exception as e:
                print(f"Failed to schedule with sequence {i}: {str(e)}")
                continue
            
            if (i+1) % 10 == 0:
                current_best = "N/A" if self.best_metrics is None else f"{self.best_metrics['avg_lateness']:.2f}"
                print(f"Completed {i+1}/{num_simulations} simulations. Current best avg lateness: {current_best}")
        
        if not valid_schedule_found:
            raise ValueError("Could not find any valid schedule. Please check the input data or adjust simulation parameters.")
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        return self.best_metrics
    
    def get_detailed_schedule(self) -> Dict:
        """
        Return a detailed description of the best schedule found.
        """
        if self.best_schedule is None:
            raise ValueError("No simulations have been run yet")
        
        # Re-run the best schedule to ensure state is correct
        self.factory.schedule_all_orders(self.best_schedule)
        
        # Detailed information for each order
        order_details = []
        for order_id in self.best_schedule:
            order = self.factory.orders[order_id]
            order_details.append({
                "order_id": order.order_id,
                "product_type": order.product_type,
                "cut_machine": order.cut_machine_id,
                "cut_start": order.cut_start_time,
                "cut_end": order.cut_end_time,
                "sew_machine": order.sew_machine_id,
                "sew_start": order.sew_start_time,
                "sew_end": order.sew_end_time,
                "pack_start": order.pack_start_time,
                "pack_end": order.pack_end_time,
                "completion_time": order.completion_time,
                "deadline": order.deadline,
                "lateness": order.lateness
            })
        
        return {
            "best_sequence": self.best_schedule,
            "metrics": self.best_metrics,
            "order_details": order_details
        }
    
    def visualize_schedule(self, save_gantt=None, save_lateness=None):
        """
        Generate and display visualizations for the schedule.
        """
        if self.best_schedule is None:
            raise ValueError("No simulations have been run yet")
        
        # Create a Gantt chart
        gantt_fig, gantt_ax = self._create_gantt_chart(save_gantt)
        
        # Create a lateness chart
        lateness_fig, lateness_ax = self._create_lateness_chart(save_lateness)
        
        return gantt_fig, lateness_fig
    
    def _create_gantt_chart(self, save_path=None):
        """
        Create a Gantt chart for the current schedule.
        """
        # Get detailed schedule
        detailed = self.get_detailed_schedule()
        order_details = detailed['order_details']
        
        # Define colors for different stages
        colors = {
            'cut': 'lightblue',
            'sew': 'lightgreen',
            'pack': 'salmon'
        }
        
        # Prepare data for Gantt chart
        machine_tasks = {}
        
        # Add all tasks to the appropriate machines
        for order in order_details:
            # Cutting task
            cut_machine = order['cut_machine']
            if cut_machine not in machine_tasks:
                machine_tasks[cut_machine] = []
            machine_tasks[cut_machine].append({
                'task': f"{order['order_id']} (Cut)",
                'start': order['cut_start'],
                'end': order['cut_end'],
                'type': 'cut',
                'product_type': order['product_type']
            })
            
            # Sewing task
            sew_machine = order['sew_machine']
            if sew_machine not in machine_tasks:
                machine_tasks[sew_machine] = []
            machine_tasks[sew_machine].append({
                'task': f"{order['order_id']} (Sew)",
                'start': order['sew_start'],
                'end': order['sew_end'],
                'type': 'sew',
                'product_type': order['product_type']
            })
            
            # Packing task
            pack_machine = 'Pack_1'  # Assuming 1 packing machine
            if pack_machine not in machine_tasks:
                machine_tasks[pack_machine] = []
            machine_tasks[pack_machine].append({
                'task': f"{order['order_id']} (Pack)",
                'start': order['pack_start'],
                'end': order['pack_end'],
                'type': 'pack',
                'product_type': order['product_type']
            })
        
        # Define machine order to show all machines
        machine_order = [
            'Cut_1', 'Cut_2',           # 2 cutting machines
            'Sew_1', 'Sew_2', 'Sew_3',  # 3 sewing machines
            'Pack_1'                     # 1 packing machine
        ]
        
        # Make sure all machines are included even if empty
        for machine_id in machine_order:
            if machine_id not in machine_tasks:
                machine_tasks[machine_id] = []
        
        # Calculate task durations for analysis
        task_durations = []
        for machine_id in machine_tasks:
            for task in machine_tasks[machine_id]:
                task_durations.append(task['end'] - task['start'])
        
        if task_durations:
            print(f"Task duration analysis:")
            print(f"  Average task duration: {np.mean(task_durations):.2f}")
            print(f"  Min task duration: {min(task_durations)}")
            print(f"  Max task duration: {max(task_durations)}")
            print(f"  Total number of tasks: {len(task_durations)}")
            print(f"  Median task duration: {np.median(task_durations):.2f}")
        
        # Create the Gantt chart
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Set up y-axis (machines)
        y_labels = machine_order  # Use all machines in order
        y_ticks = range(len(y_labels))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        
        # Draw tasks for each machine
        for machine_idx, machine in enumerate(machine_order):
            for task in machine_tasks[machine]:
                ax.barh(
                    machine_idx, 
                    task['end'] - task['start'], 
                    left=task['start'], 
                    color=colors[task['type']],
                    edgecolor='black',
                    alpha=0.8
                )
                # Add text for order ID if the task is wide enough
                if task['end'] - task['start'] > 5:  # Adjust threshold for readability
                    ax.text(
                        (task['start'] + task['end']) / 2,
                        machine_idx,
                        task['task'],
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=8
                    )
        
        # Set labels and title
        ax.set_xlabel('Time Units')
        ax.set_ylabel('Machine')
        ax.set_title('Production Schedule Gantt Chart')
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add legend
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors['cut']),
            plt.Rectangle((0, 0), 1, 1, color=colors['sew']),
            plt.Rectangle((0, 0), 1, 1, color=colors['pack'])
        ]
        ax.legend(legend_handles, ['Cutting', 'Sewing', 'Packing'], loc='upper right')
        
        # Save the chart if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the chart
        plt.tight_layout()
        return fig, ax
    
    def _create_lateness_chart(self, save_path=None):
        """
        Create a chart showing the lateness of each order.
        """
        # Get metrics
        metrics = self.best_metrics
        
        # Extract lateness data
        order_ids = []
        lateness_values = []
        
        for order_id, lateness in metrics['order_lateness'].items():
            order_ids.append(order_id)
            lateness_values.append(lateness)
        
        # Sort by lateness
        sorted_indices = np.argsort(lateness_values)[::-1]  # Descending
        sorted_orders = [order_ids[i] for i in sorted_indices]
        sorted_lateness = [lateness_values[i] for i in sorted_indices]
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the lateness values
        bars = ax.bar(range(len(sorted_orders)), sorted_lateness, 
                      color=['red' if x > 0 else 'green' for x in sorted_lateness])
        
        # Add order labels
        ax.set_xticks(range(len(sorted_orders)))
        ax.set_xticklabels(sorted_orders, rotation=90)
        
        # Add labels and title
        ax.set_xlabel('Order ID')
        ax.set_ylabel('Lateness (Time Units)')
        ax.set_title('Order Lateness')
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Save the chart if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the chart
        plt.tight_layout()
        return fig, ax
    
    def create_machine_utilization_chart(self, save_path=None):
        """Create a chart showing machine utilization percentages."""
        if not self.best_schedule:
            raise ValueError("No simulations have been run yet")

        # Get schedule details
        detailed = self.get_detailed_schedule()
        order_details = detailed['order_details']
        
        # Calculate total schedule time
        total_time = max(order['completion_time'] for order in order_details)
        
        # Calculate utilization for each machine
        machine_usage = {}
        
        # Initialize all machines with zero utilization
        for machine_id in ['Cut_1', 'Cut_2', 'Sew_1', 'Sew_2', 'Sew_3', 'Pack_1']:
            machine_type = 'Cutting' if 'Cut' in machine_id else 'Sewing' if 'Sew' in machine_id else 'Packing'
            machine_usage[machine_id] = {'busy_time': 0, 'type': machine_type}
        
        # Process all orders
        for order in order_details:
            # Add cutting machine usage
            machine_id = order['cut_machine']
            machine_usage[machine_id]['busy_time'] += order['cut_end'] - order['cut_start']
            
            # Add sewing machine usage
            machine_id = order['sew_machine']
            machine_usage[machine_id]['busy_time'] += order['sew_end'] - order['sew_start']
            
            # Add packing machine usage
            machine_id = 'Pack_1'
            machine_usage[machine_id]['busy_time'] += order['pack_end'] - order['pack_start']
        
        # Define machine order to show all machines
        machine_order = [
            'Cut_1', 'Cut_2',           # 2 cutting machines
            'Sew_1', 'Sew_2', 'Sew_3',  # 3 sewing machines
            'Pack_1'                     # 1 packing machine
        ]

        # Calculate utilization percentages
        for machine in machine_usage.values():
            machine['utilization'] = (machine['busy_time'] / total_time) * 100 if total_time > 0 else 0
        
        # Sort machines according to defined order
        machines = machine_order  # Use all machines in order
        utilization = [machine_usage[m]['utilization'] for m in machines]
        colors = ['lightblue' if machine_usage[m]['type'] == 'Cutting'
                 else 'lightgreen' if machine_usage[m]['type'] == 'Sewing'
                 else 'salmon' for m in machines]
        
        # Print utilization stats
        print(f"Machine utilization analysis:")
        print(f"  Total schedule time: {total_time}")
        print(f"  Average machine utilization: {np.mean(utilization):.2f}%")
        print(f"  Cutting machines avg: {np.mean([u for i, u in enumerate(utilization) if 'Cut' in machines[i]]):.2f}%")
        print(f"  Sewing machines avg: {np.mean([u for i, u in enumerate(utilization) if 'Sew' in machines[i]]):.2f}%")
        print(f"  Packing machines avg: {np.mean([u for i, u in enumerate(utilization) if 'Pack' in machines[i]]):.2f}%")

        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bars
        bars = ax.bar(machines, utilization, color=colors)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Customize chart
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Machine Utilization Rates')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limit to show better perspective on utilization
        ax.set_ylim(0, max(100, max(utilization) * 1.1))
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, label=l) 
                         for c, l in [('lightblue', 'Cutting'),
                                    ('lightgreen', 'Sewing'),
                                    ('salmon', 'Packing')]]
        ax.legend(handles=legend_elements)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.tight_layout()
        return fig, ax