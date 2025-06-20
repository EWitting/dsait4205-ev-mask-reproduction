import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # Import Line2D for creating proxy artists

def parse_metric(metric_str):
    """
    Parses a string like '0.96 ± 0.00' into mean and std float values.
    Returns (mean, std) or (None, None) if parsing fails.
    """
    try:
        parts = metric_str.split('±')
        mean = float(parts[0].strip())
        std = float(parts[1].strip())
        return mean, std
    except (AttributeError, IndexError, ValueError):
        return None, None

def main():
    """
    Main function to find CSVs, process data, and generate separate plots
    with a custom, multi-part descriptive legend.
    """
    # --- 1. Find and Load Data ---
    csv_files = glob.glob('./nbs/results_offset_*.csv')
    if not csv_files:
        print("Error: No 'results_offset_*.csv' files found in the './nbs/' directory.")
        return

    all_data = []
    for f in csv_files:
        match = re.search(r'results_offset_(\d+)\.csv', f)
        if match:
            offset = int(match.group(1))
            try:
                df_temp = pd.read_csv(f)
                df_temp['offset'] = offset
                all_data.append(df_temp)
            except pd.errors.EmptyDataError:
                print(f"Warning: Skipping empty file: {f}")

    if not all_data:
        print("Error: No data could be loaded from the found CSV files.")
        return

    df = pd.concat(all_data, ignore_index=True)
    
    # --- 2. Process and Clean Data ---
    for metric in ['Accuracy', 'IoU', 'AP']:
        df[[f'{metric}_mean', f'{metric}_std']] = df[metric].apply(
            lambda x: pd.Series(parse_metric(x))
        )

    df['config_name_only'] = df['train_config'].apply(lambda tc:
        "15" if "'layers': 'all'" in tc else
        "5" if "'epochs': 5" in tc else
        "2" if "'epochs': 2" in tc else
        "Unknown Config"
    )
    
    df = df.sort_values(by=['config_name_only', 'window_len', 'offset'])

    # --- 3. Define Mappings and Generate Plots ---
    
    color_map = {
        "15": "blue",
        "5": "green",
        "2": "red"
    }

    unique_windows = sorted(df['window_len'].unique())
    linestyles = ['-', '--', ':'] # Solid, dashed, dotted
    linestyle_map = {win: linestyles[i % len(linestyles)] for i, win in enumerate(unique_windows)}

    metrics_to_plot = {
        'AP': ('Average Precision (mAP)', './nbs/mAP_vs_offset.png'),
        'Accuracy': ('Accuracy', './nbs/Accuracy_vs_offset.png'),
        'IoU': ('Mean IoU', './nbs/mIoU_vs_offset.png')
    }

    for metric_key, (title, filename) in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        for config_name, config_group in df.groupby('config_name_only'):
            if config_name not in color_map:
                continue
            color = color_map[config_name]
            
            for win_len, subset in config_group.groupby('window_len'):
                linestyle = linestyle_map[win_len]
                subset = subset.sort_values('offset')
                
                # Plot the data WITHOUT a label
                ax.plot(
                    subset['offset'],
                    subset[f'{metric_key}_mean'],
                    marker='o',
                    linestyle=linestyle,
                    color=color
                )

        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Offset', fontsize=12)
        ax.set_ylabel(metric_key, fontsize=12)
        
        if metric_key == 'Accuracy':
            ax.set_ylim(0.90, 1.0)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(sorted(df['offset'].unique()))
        
        # --- Create Custom Legends ---
        
        # 1. Create proxy artists for the color legend
        color_proxies = [Line2D([0], [0], linestyle='-', color=color, marker='o', label=name)
                         for name, color in color_map.items()]
        
        # 2. Create the first legend (for colors) and add it to the plot
        color_legend = ax.legend(handles=color_proxies, title='Epochs',
                                 bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.add_artist(color_legend)

        # 3. Create proxy artists for the line style legend
        style_proxies = [Line2D([0], [0], linestyle=style, color='black', label=f'{win}ms')
                         for win, style in linestyle_map.items()]
                         
        # 4. Create the second legend (for line styles)
        # Note: We don't need to call ax.add_artist() again. The last legend is added by default.
        ax.legend(handles=style_proxies, title='Window Length',
                  bbox_to_anchor=(1.02, 0.65), loc='upper left')

        # --- End Custom Legends ---

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully generated and saved: {filename}")

if __name__ == "__main__":
    main()