import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sika_manufacturing_data(n_batches=2000):
    """
    Generate realistic Sika cementitious manufacturing data
    Based on Welwyn Garden City site operations
    """
    
    # Define product families and their characteristics
    product_families = {
        'fine_white': {
            'color_variants': ['white', 'light', 'off-white', 'cream'],
            'base_products': ['SikaTop-121', 'SikaTop-122', 'SikaGrout-212'],
            'flush_group': 'light_fine'
        },
        'coarse_grey': {
            'color_variants': ['grey', 'gray', 'concrete', 'natural'],
            'base_products': ['SikaGrout-214', 'SikaTop-132', 'SikaRep-212'],
            'flush_group': 'grey_coarse'
        },
        'dark': {
            'color_variants': ['dark', 'black', 'charcoal', 'anthracite'],
            'base_products': ['SikaTop-141', 'SikaColor-201', 'SikaTop-142'],
            'flush_group': 'dark'
        },
        'fine_dark': {
            'color_variants': ['dark_fine', 'fine_black', 'dark', 'black_fine'],
            'base_products': ['SikaTop-151', 'SikaRep-221', 'SikaTop-152'],
            'flush_group': 'dark_fine'
        },
        'coarse_white': {
            'color_variants': ['white_coarse', 'coarse_light', 'white', 'light_coarse'],
            'base_products': ['SikaGrout-315', 'SikaTop-162', 'SikaRep-315'],
            'flush_group': 'light_coarse'
        }
    }
    
    # Generate manufacturing batches
    batches = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_batches):
        # Select random product family
        true_family = random.choice(list(product_families.keys()))
        family_info = product_families[true_family]
        
        # Generate product name
        base_product = random.choice(family_info['base_products'])
        
        # Generate batch start time (working hours: 6 AM to 10 PM)
        days_offset = random.randint(0, 120)  # 4 months of data
        hour = random.randint(6, 22)
        minute = random.randint(0, 59)
        batch_start = start_date + timedelta(days=days_offset, hours=hour, minutes=minute)
        
        # Generate labeled family (sometimes incorrect due to human error)
        if random.random() < 0.15:  # 15% chance of mislabeling
            labeled_family = random.choice(list(product_families.keys()))
        else:
            labeled_family = true_family
        
        # Generate labeled color (sometimes inconsistent)
        true_color_variants = family_info['color_variants']
        if random.random() < 0.20:  # 20% chance of using different color variant
            # Use variant from different family (human confusion)
            all_variants = []
            for fam in product_families.values():
                all_variants.extend(fam['color_variants'])
            labeled_color = random.choice(all_variants)
        else:
            labeled_color = random.choice(true_color_variants)
        
        # Generate batch size (kg)
        batch_size = np.random.normal(1500, 300)  # Normal distribution around 1500kg
        batch_size = max(500, min(3000, batch_size))  # Clamp between 500-3000kg
        
        # Generate processing time (minutes)
        base_time = 45  # Base processing time
        size_factor = (batch_size - 1000) / 1000 * 10  # Larger batches take longer
        processing_time = base_time + size_factor + np.random.normal(0, 5)
        processing_time = max(20, processing_time)
        
        # Determine if flush is actually needed (ground truth)
        # This would be based on true product compatibility
        true_flush_group = family_info['flush_group']
        
        # For the first batch, no flush needed
        if i == 0:
            flush_needed = False
            previous_flush_group = true_flush_group
        else:
            # Flush needed if switching between incompatible groups
            flush_needed = previous_flush_group != true_flush_group
            previous_flush_group = true_flush_group
        
        # Generate flush time if needed (10-30 minutes)
        flush_time = np.random.uniform(10, 30) if flush_needed else 0
        
        # Create batch record
        batch = {
            'batch_id': f'WGC_{i:05d}',
            'product_name': base_product,
            'batch_start_time': batch_start,
            'labeled_family': labeled_family,
            'labeled_color': labeled_color,
            'true_family': true_family,  # This would be unknown in real scenario
            'true_flush_group': true_flush_group,  # This would be unknown in real scenario
            'batch_size_kg': round(batch_size),
            'processing_time_min': round(processing_time, 1),
            'flush_needed': flush_needed,
            'flush_time_min': round(flush_time, 1),
            'total_time_min': round(processing_time + flush_time, 1)
        }
        
        batches.append(batch)
    
    return pd.DataFrame(batches)

def add_manufacturing_metrics(df):
    """
    Add manufacturing efficiency metrics
    """
    # Calculate current efficiency (with suboptimal scheduling)
    total_processing_time = df['processing_time_min'].sum()
    total_flush_time = df['flush_time_min'].sum()
    total_time = df['total_time_min'].sum()
    
    current_efficiency = total_processing_time / total_time
    flush_percentage = total_flush_time / total_time
    
    # Calculate number of flushes
    num_flushes = df['flush_needed'].sum()
    
    return {
        'total_batches': len(df),
        'total_time_hours': round(total_time / 60, 1),
        'processing_time_hours': round(total_processing_time / 60, 1),
        'flush_time_hours': round(total_flush_time / 60, 1),
        'current_efficiency': round(current_efficiency, 3),
        'flush_percentage': round(flush_percentage * 100, 1),
        'number_of_flushes': num_flushes,
        'avg_flush_time_min': round(df[df['flush_needed']]['flush_time_min'].mean(), 1)
    }

def main():
    print("Generating Sika Welwyn Garden City manufacturing data...")
    
    # Generate the dataset
    df = generate_sika_manufacturing_data(2000)
    
    # Sort by batch start time (chronological order)
    df = df.sort_values('batch_start_time').reset_index(drop=True)
    
    # Save full dataset (including ground truth for training)
    df.to_csv('sika_manufacturing_data.csv', index=False)
    
    # Create operational dataset (without ground truth - what we'd actually have)
    operational_df = df.drop(['true_family', 'true_flush_group'], axis=1)
    operational_df.to_csv('sika_operational_data.csv', index=False)
    
    # Calculate and display metrics
    metrics = add_manufacturing_metrics(df)
    
    print(f"\nGenerated {len(df)} manufacturing batches")
    print("\nDataset overview:")
    print(df[['batch_id', 'product_name', 'labeled_family', 'labeled_color', 'flush_needed']].head(10))
    
    print(f"\nManufacturing Efficiency Metrics:")
    print(f"Total batches: {metrics['total_batches']}")
    print(f"Total time: {metrics['total_time_hours']} hours")
    print(f"Processing time: {metrics['processing_time_hours']} hours")
    print(f"Flush time: {metrics['flush_time_hours']} hours")
    print(f"Current efficiency: {metrics['current_efficiency']} ({metrics['flush_percentage']}% time spent on flushing)")
    print(f"Number of flushes: {metrics['number_of_flushes']}")
    print(f"Average flush time: {metrics['avg_flush_time_min']} minutes")
    
    print(f"\nLabel consistency analysis:")
    family_consistency = (df['labeled_family'] == df['true_family']).mean()
    print(f"Family labeling accuracy: {family_consistency:.1%}")
    
    print(f"\nColor variant distribution:")
    print(df['labeled_color'].value_counts().head(10))
    
    print("\nData saved to:")
    print("- 'sika_manufacturing_data.csv' (with ground truth for training)")
    print("- 'sika_operational_data.csv' (operational data without ground truth)")

if __name__ == "__main__":
    main()