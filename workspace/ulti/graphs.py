import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def process_csv(file_name, metric):
    df = pd.read_csv(file_name)
    # Convert the string representation of dictionary to actual dictionary
    df['best_data'] = df['best_data'].apply(ast.literal_eval)
    # Extract the metric of interest and the circuit label
    df[metric] = df['best_data'].apply(lambda x: x[metric])
    df['circuit_label'] = df['circuit label'].astype(int)  # Ensure circuit label is integer for color mapping
    return df[['circuit_label', metric]]

# Function to compare a single metric between two data files with color coding for circuit labels using Seaborn
def compare_single_metric(metric, file1, file2, x_label, y_label, plot_title):
    df1 = process_csv(file1, metric)
    df2 = process_csv(file2, metric)
    
    # Merge the two dataframes on circuit label
    merged_df = pd.merge(df1, df2, on='circuit_label', suffixes=('_file1', '_file2'))
    
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Create a scatter plot to compare the metrics using Seaborn
    scatter = sns.scatterplot(x=f'{metric}_file1', y=f'{metric}_file2', 
                         hue='circuit_label', data=merged_df, 
                         palette='viridis', alpha=0.7)
    
    # Set the scale to logarithmic
    plt.xscale('log')
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    
    # Plot a reference line where the metrics would be equal in both files
    min_val = min(merged_df[f'{metric}_file1'].min(), merged_df[f'{metric}_file2'].min())
    max_val = max(merged_df[f'{metric}_file1'].max(), merged_df[f'{metric}_file2'].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
    
    # Create a color bar for the circuit labels
    norm = plt.Normalize(merged_df['circuit_label'].min(), merged_df['circuit_label'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=scatter.axes)
    cbar.set_label('Circuit Label')
    
    # Show the plot with a grid
    plt.grid(True)
    plt.show()