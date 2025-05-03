import os
import pandas as pd
import matplotlib.pyplot as plt

# Number of decimal places to round numeric values (excluding learning rate & train loss)
DECIMALS = 2

def csv_to_image(csv_path, output_path, decimals=DECIMALS):
    """
    Read a CSV file, round numeric columns (except learning rate & train loss), and export it as a table image.
    """
    df = pd.read_csv(csv_path)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Exclude columns with 'learning rate' or 'train loss' in the name from rounding
    skip_cols = [
        c for c in numeric_cols
        if ('learning' in c.lower() and 'rate' in c.lower()) 
        or ('train' in c.lower() and 'loss' in c.lower())
    ]
    round_cols = [c for c in numeric_cols if c not in skip_cols]
    
    # Round selected numeric columns
    df[round_cols] = df[round_cols].round(decimals)
    
    # Determine figure size based on table dimensions
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2, max(len(df) * 0.5, 2)))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def process_logs_folder(logs_dir, output_dir, decimals=DECIMALS):
    """
    Process all CSV files in `logs_dir`, rounding numbers (except learning rate & train loss)
    and exporting each as a PNG in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(logs_dir):
        if fname.lower().endswith('.csv'):
            csv_path = os.path.join(logs_dir, fname)
            image_name = os.path.splitext(fname)[0] + '.png'
            output_path = os.path.join(output_dir, image_name)
            csv_to_image(csv_path, output_path, decimals)
            print(f"Generated rounded image (excl. LR & Train Loss): {output_path}")

if __name__ == "__main__":
    logs_dir = "logs"           # Folder containing your CSV log files
    output_dir = "table_images" # Folder where PNGs will be saved
    process_logs_folder(logs_dir, output_dir, DECIMALS)
