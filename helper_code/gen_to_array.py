import pandas as pd
import numpy as np
from scipy.io import savemat
import sys

def csv_to_mat(csv_file_path, mat_file_path='training_data.mat'):
    """
    Convert CSV training data to MATLAB .mat file with sequential iteration numbers
    
    Creates two arrays:
    - iteration: [1, 2, 3, 4, ..., N] where N is total number of data points
    - loss: [loss1, loss2, loss3, ..., lossN] corresponding loss values
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} data points from {csv_file_path}")
        
        # Check if required columns exist
        if 'loss' not in df.columns:
            print("Error: 'loss' column not found in CSV file")
            return None
            
        # Extract loss values
        loss_values = df['loss'].values
        
        # Create sequential iteration numbers starting from 1
        iteration_numbers = np.arange(1, len(loss_values) + 1)
        
        # Create the data dictionary for MATLAB
        matlab_data = {
            'iteration': iteration_numbers,
            'loss': loss_values,
            'epoch': df['epoch'].values if 'epoch' in df.columns else None,
            'original_iteration': df['iteration'].values if 'iteration' in df.columns else None
        }
        
        # Remove None values from the dictionary
        matlab_data = {k: v for k, v in matlab_data.items() if v is not None}
        
        # Save to .mat file
        savemat(mat_file_path, matlab_data)
        
        print(f"Successfully saved MATLAB file: {mat_file_path}")
        print(f"Arrays created:")
        print(f"  - iteration: {len(iteration_numbers)} elements [1 to {len(iteration_numbers)}]")
        print(f"  - loss: {len(loss_values)} elements")
        
        if 'epoch' in matlab_data:
            epochs = np.unique(df['epoch'])
            print(f"  - epoch: {len(epochs)} unique epochs ({epochs.min()} to {epochs.max()})")
            
        # Display statistics
        print(f"\nLoss Statistics:")
        print(f"  Min loss: {np.min(loss_values):.6f}")
        print(f"  Max loss: {np.max(loss_values):.6f}")
        print(f"  Mean loss: {np.mean(loss_values):.6f}")
        print(f"  Std loss: {np.std(loss_values):.6f}")
        
        # Show data per epoch
        if 'epoch' in df.columns:
            print(f"\nData points per epoch:")
            epoch_counts = df['epoch'].value_counts().sort_index()
            for epoch, count in epoch_counts.items():
                print(f"  Epoch {epoch}: {count} points")
        
        return matlab_data
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def create_matlab_script(mat_file_path='training_data.mat', script_path='plot_training.m'):
    """
    Create a MATLAB script to load and plot the data
    """
    matlab_script = f"""% MATLAB script to load and plot training data
% Generated automatically


% Load the data
load('{mat_file_path}');

% Basic plot
figure(1);
plot(iteration, loss, 'b-', 'LineWidth', 1);
xlabel('Iteration');
ylabel('Loss');
title('Training Loss vs Iteration');
grid on;

% If epoch data exists, create epoch-colored plot
if exist('epoch', 'var')
    figure(2);
    scatter(iteration, loss, 10, epoch, 'filled');
    colorbar;
    xlabel('Iteration');
    ylabel('Loss');
    title('Training Loss by Epoch');
    colormap(jet);
    
    % Add epoch boundaries
    hold on;
    unique_epochs = unique(epoch);
    for i = 2:length(unique_epochs)
        epoch_start = find(epoch == unique_epochs(i), 1);
        xline(epoch_start, 'r--', 'Alpha', 0.7);
    end
    hold off;
end

% Display statistics
fprintf('Training Data Statistics:\\n');
fprintf('Total iterations: %d\\n', length(iteration));
fprintf('Loss range: %.6f to %.6f\\n', min(loss), max(loss));
fprintf('Mean loss: %.6f\\n', mean(loss));

if exist('epoch', 'var')
    fprintf('Epochs: %d to %d\\n', min(epoch), max(epoch));
end
"""
    
    try:
        with open(script_path, 'w') as f:
            f.write(matlab_script)
        print(f"MATLAB script created: {script_path}")
    except Exception as e:
        print(f"Error creating MATLAB script: {e}")

if __name__ == "__main__":
    # Get file paths
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
        mat_file = sys.argv[2] if len(sys.argv) >= 3 else "training_data.mat"
    else:
        csv_file = input("Enter CSV file path (default: Loss_extract.txt): ").strip()
        if not csv_file:
            csv_file = "Loss_extract.txt"
            
        mat_file = input("Enter output .mat file name (default: training_data.mat): ").strip()
        if not mat_file:
            mat_file = "training_data.mat"
    
    print(f"Converting {csv_file} to {mat_file}")
    print("-" * 50)
    
    # Convert CSV to MAT
    result = csv_to_mat(csv_file, mat_file)
    
    if result is not None:
        # Ask if user wants MATLAB script
        create_script = input("\nCreate MATLAB plotting script? (y/n): ").strip().lower()
        if create_script in ['y', 'yes']:
            script_name = input("Enter script name (default: plot_training.m): ").strip()
            if not script_name:
                script_name = "plot_training.m"
            create_matlab_script(mat_file, script_name)
        
        print(f"\nFiles created:")
        print(f"  - {mat_file} (MATLAB data file)")
        if create_script in ['y', 'yes']:
            print(f"  - {script_name} (MATLAB plotting script)")
        
        print(f"\nTo use in MATLAB:")
        print(f"  1. load('{mat_file}')")
        print(f"  2. plot(iteration, loss)")
        print(f"  3. Or run: {script_name if create_script in ['y', 'yes'] else 'plot_training.m'}")