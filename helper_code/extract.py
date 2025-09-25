import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

def extract_training_data(log_file_path, output_file_path, sample_per_epoch=100):
    """
    Extract epoch, iteration, and loss from MTR training log
    """
    extracted_data = []
    
    # Pattern to match training log lines with loss
    # Matches: epoch: X/30, acc_iter=XXXX, ... loss=XXX.XXX
    pattern = r'epoch: (\d+)/\d+.*?acc_iter=(\d+).*?loss=([+-]?\d+\.?\d*)'
    
    try:
        with open(log_file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                match = re.search(pattern, line)
                if match:
                    epoch = int(match.group(1))
                    acc_iter = int(match.group(2))
                    loss_value = float(match.group(3))
                    extracted_data.append((epoch, acc_iter, loss_value))
        
        if not extracted_data:
            print("No training data found in the log file.")
            print("Please check if the log file format matches the expected pattern.")
            return None
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(extracted_data, columns=['epoch', 'iteration', 'loss'])
        
        # Optional: Sample data if there are too many points per epoch
        if sample_per_epoch and sample_per_epoch > 0:
            sampled_data = []
            for epoch in df['epoch'].unique():
                epoch_data = df[df['epoch'] == epoch]
                
                if len(epoch_data) > sample_per_epoch:
                    # Sample evenly across the epoch
                    indices = range(0, len(epoch_data), len(epoch_data) // sample_per_epoch)
                    epoch_sampled = epoch_data.iloc[list(indices)[:sample_per_epoch]]
                else:
                    epoch_sampled = epoch_data
                    
                sampled_data.append(epoch_sampled)
            
            df = pd.concat(sampled_data, ignore_index=True)
        
        # Save to CSV format (even with .txt extension for compatibility)
        df.to_csv(output_file_path, index=False)
        
        print(f"Extracted {len(df)} data points from {len(df['epoch'].unique())} epochs")
        print(f"Data saved to: {output_file_path}")
        
        # Display statistics by epoch
        print("\nData points per epoch:")
        epoch_counts = df['epoch'].value_counts().sort_index()
        for epoch, count in epoch_counts.items():
            epoch_data = df[df['epoch'] == epoch]
            print(f"  Epoch {epoch}: {count} points, Loss range: {epoch_data['loss'].min():.3f} - {epoch_data['loss'].max():.3f}")
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total iterations: {df['iteration'].min()} - {df['iteration'].max()}")
        print(f"  Loss range: {df['loss'].min():.3f} - {df['loss'].max():.3f}")
        print(f"  Initial loss: {df['loss'].iloc[0]:.3f}")
        print(f"  Final loss: {df['loss'].iloc[-1]:.3f}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Could not find log file: {log_file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def plot_training_progress(df, output_plot_path=None):
    """
    Create a loss vs iteration plot with epoch markers
    """
    if df is None:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot loss vs iteration, colored by epoch
    epochs = df['epoch'].unique()
    colors = plt.cm.tab10(range(len(epochs)))
    
    for i, epoch in enumerate(sorted(epochs)):
        epoch_data = df[df['epoch'] == epoch]
        plt.plot(epoch_data['iteration'], epoch_data['loss'], 
                'o-', color=colors[i % len(colors)], markersize=2, linewidth=1,
                label=f'Epoch {epoch}', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Progress')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add epoch boundaries as vertical lines
    for epoch in epochs[1:]:  # Skip first epoch
        first_iter = df[df['epoch'] == epoch]['iteration'].min()
        plt.axvline(x=first_iter, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_plot_path}")
    
    plt.show()

def analyze_convergence(df):
    """
    Analyze training convergence patterns
    """
    if df is None:
        return
        
    print("\nConvergence Analysis:")
    print("-" * 50)
    
    for epoch in sorted(df['epoch'].unique()):
        epoch_data = df[df['epoch'] == epoch].copy()
        
        if len(epoch_data) > 1:
            # Calculate loss improvement within epoch
            initial_loss = epoch_data['loss'].iloc[0]
            final_loss = epoch_data['loss'].iloc[-1]
            improvement = initial_loss - final_loss
            improvement_pct = (improvement / initial_loss) * 100 if initial_loss != 0 else 0
            
            print(f"Epoch {epoch}:")
            print(f"  Initial loss: {initial_loss:.3f}")
            print(f"  Final loss: {final_loss:.3f}")
            print(f"  Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
            print(f"  Min loss: {epoch_data['loss'].min():.3f}")
            print(f"  Max loss: {epoch_data['loss'].max():.3f}")
            print()

# Usage
if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) >= 2:
        log_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else "training_progress.csv"
        sample_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    else:
        # Interactive input
        log_file = input("Enter the path to your log file: ").strip()
        if not log_file:
            log_file = r"C:\Test_Waymo\log_data\log_train_20250920-epoch0.txt"  # Default
        
        output_file = input("Enter output file name (default: Loss_extract.txt): ").strip()
        if not output_file:
            output_file = "Loss_extract.txt"
            
        sample_input = input("Enter max points per epoch (default: 100, 0 for all): ").strip()
        sample_size = int(sample_input) if sample_input else 100
    
    print(f"Processing log file: {log_file}")
    print(f"Output CSV: {output_file}")
    print(f"Max points per epoch: {sample_size if sample_size > 0 else 'All'}")
    print("-" * 50)
    
    # Extract data
    df = extract_training_data(log_file, output_file, sample_size)
    
    if df is not None:
        # Analyze convergence
        analyze_convergence(df)
        
        # Ask if user wants to plot
        plot_choice = input("\nDo you want to create a plot? (y/n): ").strip().lower()
        if plot_choice in ['y', 'yes']:
            plot_path = input("Enter plot file name (optional, press Enter to just display): ").strip()
            plot_path = plot_path if plot_path else None
            plot_training_progress(df, plot_path)
        
        print(f"\nData successfully extracted to {output_file}")
        print("You can now use this file with any plotting tool or import it into Excel/Python for analysis.")
        print("Format: epoch,iteration,loss (CSV format even with .txt extension)")