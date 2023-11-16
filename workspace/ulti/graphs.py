import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import ast
import os


def process_csv(file_name, metric):
    df = pd.read_csv(file_name)
    # Convert the string representation of dictionary to actual dictionary
    df['best_data'] = df['best_data'].apply(ast.literal_eval)
    # Extract the metric of interest and the circuit label
    df[metric] = df['best_data'].apply(lambda x: x[metric])
    df['circuit_label'] = df['circuit label'].astype(int)  # Ensure circuit label is integer for color mapping
    return df[['circuit_label', metric]]

# Function to compare a single metric between two data files with color coding for circuit labels using Seaborn
def compare_single_metric(metric, file1, file2, x_label, y_label, plot_title, labelling):
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
    summary_table = pd.DataFrame(average_metrics)
    summary_table['% Difference from Baseline'] = ((summary_table[metric] - baseline) / baseline) * 100

    return summary_table



def analyze_lookahead_times(directory):
    # Function to extract and calculate average time from a file
    def extract_average_time(file_path):
        df = pd.read_csv(file_path)
        time_values = df['best_data'].apply(lambda x: eval(x)['time'])
        return time_values.mean()

    # Function to define the exponential curve
    def exponential_curve(x, a, b):
        return a * np.exp(b * x)

    # Dictionary to store average times for each file
    average_times = {}

    # Looping through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            lookahead_number = int(filename.split('_')[-1].split('.')[0])
            average_times[lookahead_number] = extract_average_time(file_path)

    # Creating a DataFrame for plotting and summarizing
    average_time_df = pd.DataFrame(list(average_times.items()), columns=['Lookahead Number', 'Average Time'])
    sorted_average_time_df = average_time_df.sort_values('Lookahead Number')

    # Extracting x and y data for curve fitting
    x_data = sorted_average_time_df['Lookahead Number']
    y_data = sorted_average_time_df['Average Time']

    # Performing the curve fit
    params, params_covariance = curve_fit(exponential_curve, x_data, y_data, maxfev=10000)

    # Generating y values from the fitted curve
    fitted_y_data = exponential_curve(x_data, *params)

    # Plotting the original data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data Points', color='blue')
    plt.plot(x_data, fitted_y_data, label='Fitted Curve', color='red', linewidth=2)
    plt.title('Exponential Curve Fit for Time vs Lookahead Number')
    plt.xlabel('Lookahead Number')
    plt.ylabel('Average Time')
    plt.legend()
    plt.show()

    # Adding a column to show increase in time from the previous lookahead number
    sorted_average_time_df['Percentage Increase in Time'] = sorted_average_time_df['Average Time'].pct_change() * 100

    # Printing the parameters of the fitted curve and the summary table
    print('Parameters of the fitted exponential curve:', params)
    print('\nSummary Table:')
    print(sorted_average_time_df)