# Monte Carlo Simulation Results

This directory contains the output files from the Monte Carlo Apparel Production Scheduler.

## Files

- **order_schedule.csv**: Detailed schedule for each order, including start/end times for each production stage
- **best_sequence.txt**: The sequence of orders that produced the best schedule
- **summary.txt**: Summary metrics for the schedule performance
- **schedule_gantt.png**: Gantt chart visualisation of the production schedule
- **schedule_lateness.png**: Chart showing the lateness of each order
- **machine_utilization.png**: Chart showing the utilisation rates of each machine

## Interpreting Results

### Performance Metrics

- **Total Orders**: Number of orders scheduled
- **Orders Completed On Time**: Number and percentage of orders that met their deadlines
- **Average Lateness**: Average lateness across all orders
- **Maximum Lateness**: The lateness of the most delayed order
- **Order Lateness**: Breakdown of lateness by individual order

### Schedule Analysis

The Gantt chart provides a visual representation of machine utilisation and order flow. Look for:
- Idle time on machines (gaps in the schedule)
- Bottlenecks (machines that are consistently busy)
- Sequential dependencies (observe how orders flow from cutting to sewing to packing)

### Potential Improvements

If you're seeing high lateness values, consider:
1. Adjusting order priorities based on deadlines and processing times
2. Adding more capacity to bottleneck stages
3. Re-allocating orders across time periods
4. Reducing setup times by better product type batching
