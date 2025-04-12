import pandas as pd
import os

def analyze_schedule():
    """Analyze schedule for overlaps and sequential violations."""
    # Load the schedule
    schedule_path = os.path.join(os.path.dirname(__file__), "order_schedule.csv")
    df = pd.read_csv(schedule_path)

    def check_stage_overlaps(stage_df, stage_name):
        overlaps = []
        for machine in stage_df['machine'].unique():
            machine_tasks = stage_df[stage_df['machine'] == machine].sort_values('start')
            
            for i in range(len(machine_tasks)-1):
                current = machine_tasks.iloc[i]
                next_task = machine_tasks.iloc[i+1]
                
                if current['end'] > next_task['start']:
                    overlaps.append({
                        'stage': stage_name,
                        'machine': machine,
                        'order1': current['order_id'],
                        'order2': next_task['order_id'],
                        'overlap': f"{current['end']} > {next_task['start']}"
                    })
        return overlaps

    # Prepare data for each stage with correct column names
    cutting = pd.DataFrame({
        'machine': df['cut_machine'],
        'order_id': df['order_id'],
        'start': df['cut_start'],
        'end': df['cut_end']
    })

    sewing = pd.DataFrame({
        'machine': df['sew_machine'],
        'order_id': df['order_id'],
        'start': df['sew_start'],
        'end': df['sew_end']
    })

    packing = pd.DataFrame({
        'machine': df.apply(lambda x: 'Pack_1', axis=1),  # All orders use Pack_1
        'order_id': df['order_id'],
        'start': df['pack_start'],
        'end': df['pack_end']
    })

    # Check overlaps for each stage
    all_overlaps = []
    all_overlaps.extend(check_stage_overlaps(cutting, 'Cutting'))
    all_overlaps.extend(check_stage_overlaps(sewing, 'Sewing'))
    all_overlaps.extend(check_stage_overlaps(packing, 'Packing'))

    # Display results
    if all_overlaps:
        print("\nOverlaps Found:")
        print("="*80)
        overlap_df = pd.DataFrame(all_overlaps)
        print(overlap_df.to_string(index=False))
        print(f"\nTotal overlaps: {len(all_overlaps)}")
    else:
        print("\nNo overlaps found - Schedule is conflict-free!")

    # Check sequential constraints
    sequential_violations = []
    for _, order in df.iterrows():
        if order['sew_start'] < order['cut_end']:
            sequential_violations.append({
                'order_id': order['order_id'],
                'violation': 'Sewing starts before cutting ends',
                'details': f"cut_end={order['cut_end']} > sew_start={order['sew_start']}"
            })
        if order['pack_start'] < order['sew_end']:
            sequential_violations.append({
                'order_id': order['order_id'],
                'violation': 'Packing starts before sewing ends',
                'details': f"sew_end={order['sew_end']} > pack_start={order['pack_start']}"
            })

    if sequential_violations:
        print("\nSequential Constraint Violations:")
        print("="*80)
        violations_df = pd.DataFrame(sequential_violations)
        print(violations_df.to_string(index=False))
        print(f"\nTotal violations: {len(sequential_violations)}")

if __name__ == "__main__":
    analyze_schedule()
