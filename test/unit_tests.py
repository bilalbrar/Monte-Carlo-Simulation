import pytest
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apparel_scheduler import Order, Machine, Factory

def test_order_class():
    """Test the Order class functionality."""
    order = Order("O001", "A", 10, 20, 5, 100, False)
    
    # Test initial state
    assert order.order_id == "O001"
    assert order.product_type == "A"
    assert order.cut_time == 10
    assert order.sew_time == 20
    assert order.pack_time == 5
    assert order.deadline == 100
    assert order.requires_delay is False
    
    # Test scheduling state
    assert order.is_scheduled() is False
    
    # Schedule the order manually
    order.cut_start_time = 0
    order.cut_end_time = 10
    order.sew_start_time = 10
    order.sew_end_time = 30
    order.pack_start_time = 30
    order.pack_end_time = 35
    order.completion_time = 35
    
    # Test updated state
    assert order.is_scheduled() is True
    
    # Test lateness calculation
    assert order.calculate_lateness() == 0
    
    # Test with lateness
    order.completion_time = 120
    assert order.calculate_lateness() == 20


def test_machine_class():
    """Test the Machine class functionality."""
    machine = Machine("Cut_1", "cut")
    
    # Test initial state
    assert machine.machine_id == "Cut_1"
    assert machine.stage == "cut"
    assert machine.schedule == []
    
    # Test availability
    available, next_time = machine.is_available(0, 10)  # Added duration argument
    assert available is True
    assert next_time == 0
    
    # Add a task
    machine.add_task(10, 20, "O001", "A")
    
    # Test last product type
    assert machine.last_product_type == "A"
    
    # Test availability after adding task
    available, next_time = machine.is_available(15, 5)  # Added duration argument
    assert available is False  # Machine is busy during this time
    assert next_time == 20  # Next available time is after the current task ends
    
    # Test availability for a time slot after the current task
    available, next_time = machine.is_available(20, 5)
    assert available is True
    assert next_time == 20
    
    # Test setup time calculation
    assert machine.get_setup_time("A") == 0  # Same product type
    assert machine.get_setup_time("B") == 10  # Different product type
    
    # Add another task
    machine.add_task(30, 40, "O002", "B")
    
    # Test that schedule is sorted
    assert machine.schedule == [(10, 20, "O001"), (30, 40, "O002")]


def test_factory_simple_schedule():
    """Test factory scheduling with a simple case."""
    factory = Factory()  # Single machine per stage
    
    # Create a simple order
    order = Order("O001", "A", 10, 20, 5, 100, False)
    factory.add_order(order)
    
    # Schedule the order
    factory.schedule_order(order)
    
    # Check the result
    assert order.cut_machine_id == "Cut_1"
    assert order.sew_machine_id == "Sew_1"
    assert order.cut_start_time == 0
    assert order.cut_end_time == 10
    assert order.sew_start_time == 10
    assert order.sew_end_time == 30
    assert order.pack_start_time == 30
    assert order.pack_end_time == 35
    assert order.completion_time == 35
    assert order.lateness == 0


def test_factory_with_delay():
    """Test factory scheduling with post-cutting delay."""
    factory = Factory()  # Single machine per stage
    
    # Create an order that requires delay
    order = Order("O001", "A", 10, 20, 5, 100, True)
    factory.add_order(order)
    
    # Schedule the order
    factory.schedule_order(order)
    
    # Check the result (should include 48 time units delay)
    assert order.cut_start_time == 0
    assert order.cut_end_time == 10
    assert order.sew_start_time == 10 + 48  # Cut end + delay
    assert order.sew_end_time == 10 + 48 + 20
    assert order.pack_start_time == 10 + 48 + 20
    assert order.pack_end_time == 10 + 48 + 20 + 5
    assert order.completion_time == 10 + 48 + 20 + 5
    assert order.lateness == 0  # Still on time
