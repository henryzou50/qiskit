""" This module contains functions to generate graphs and tables for the results of the experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os


def process_csv(file_name, metric):
    """ Function to process a csv file and extract the metric of interest and the circuit label.
    
    Args:
        file_name (str): Name of the csv file
        metric (str): Metric of interest
    Returns:
        df (pandas.DataFrame): Dataframe with the metric and circuit label
    """
    df = pd.read_csv(file_name)
    # Convert the string representation of dictionary to actual dictionary
    df['best_data'] = df['best_data'].apply(ast.literal_eval)
    # Extract the metric of interest and the circuit label
    df[metric] = df['best_data'].apply(lambda x: x[metric])
    df['circuit_label'] = df['circuit label'].astype(int)  
    return df[['circuit_label', metric]]


def compare_single_metric(metric, file1, file2, x_label, y_label, plot_title, labelling):
    """ Function to compare a single metric between two files.
    
    Args:
        metric (str): Metric of interest
        file1 (str): Name of the first csv file
        file2 (str): Name of the second csv file
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
        plot_title (str): Title for the plot
        labelling (str): Label for the color bar
    Returns:
        None
    """
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
    cbar.set_label(labelling)
    
    # Show the plot with a grid
    plt.grid(True)
    plt.show()


def analyze_metric(data_dir, metric, baseline_filename, plot_title, x_axis_label):
    """ Function to analyze a metric across multiple files.

    Args:
        data_dir (str): Path to the directory containing the csv files
        metric (str): Metric of interest
        baseline_filename (str): Name of the file to use as baseline
        plot_title (str): Title for the plot
        x_axis_label (str): Label for the x-axis
    Returns:
        sum_table (pandas.DataFrame): A dataframe containing the average of the metric for each
        file and the percentage difference compared to the baseline
    """
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []

    for file in data_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        df['best_data'] = df['best_data'].apply(eval)
        df[metric] = df['best_data'].apply(lambda x: x.get(metric, None))
        df['file'] = file  # Add this to identify the file later

        all_data.append(df)

    if not all_data:
        print("No data to concatenate. Check if files are read correctly.")
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    # Visualization using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='circuit label', y=metric, hue='file', data=combined_data, marker='o')
    plt.title(plot_title)
    plt.xlabel(x_axis_label)
    plt.xticks(rotation=45)
    plt.legend(title='File', loc='upper left')
    plt.show()

    # Calculate average of the metric for each file
    average_metrics = combined_data.groupby('file')[metric].mean()

    # Calculate percentage difference compared to baseline
    baseline = average_metrics[baseline_filename]
    sum_table = pd.DataFrame(average_metrics)
    sum_table['% Difference from Baseline'] = ((sum_table[metric] - baseline) / baseline) * 100

    return sum_table

def plot_data(filename, y_key, title, x_label, y_label, baseline=None, point_color='forestgreen'):
    # Load the CSV file
    df = pd.read_csv(filename)

    # Process the 'best_data' column and extract the specified y_key
    df['best_data'] = df['best_data'].apply(lambda x: ast.literal_eval(x))
    df[y_key] = df['best_data'].apply(lambda x: x.get(y_key))

    # Create a new column for row numbers
    df['RowNumber'] = range(1, len(df) + 1)

    # Plotting
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df, x='RowNumber', y=y_key, color=point_color, palette="viridis", s=100, edgecolor="w", alpha=0.7)

    # Add baseline if specified
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Sabre 0.20: {baseline} {y_label}')

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    if baseline is not None:
        plt.legend(loc='upper right')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()



