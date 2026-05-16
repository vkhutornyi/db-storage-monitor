import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def extract_date_from_filename(filename):
    """Extracts date from filename (e.g., TableInformation-2026-05-15.xlsx)"""
    match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    if match:
        return datetime.strptime(match.group(0), "%Y-%m-%d")
    return None

def scan_input_folder(folder_path):
    """Scans the folder and sorts files chronologically by date in filename"""
    files_data = []
    if not os.path.exists(folder_path):
        return []
    for file in os.listdir(folder_path):
        if file.endswith(('.xlsx', '.xls', '.csv')):
            date = extract_date_from_filename(file)
            if date:
                files_data.append({'path': os.path.join(folder_path, file), 'date': date, 'name': file})
    return sorted(files_data, key=lambda x: x['date'])

def calculate_sizes(file_info):
    """Reads file and calculates total Data + Index size grouped by Company"""
    if file_info['path'].endswith('.csv'):
        df = pd.read_csv(file_info['path'])
    else:
        df = pd.read_excel(file_info['path'])
    
    # Handle empty company names by grouping them into 'System / Shared'
    df['Company Name'] = df['Company Name'].fillna('System / Shared')
    if hasattr(df['Company Name'], 'str'):
        df['Company Name'] = df['Company Name'].str.strip()
    
    # Identify size columns (Data Size + Index Size if available, fallback to Size or index)
    data_col = 'Data Size (KB)' if 'Data Size (KB)' in df.columns else None
    idx_col = 'Index Size (KB)' if 'Index Size (KB)' in df.columns else None
    size_col = 'Size (KB)' if 'Size (KB)' in df.columns else df.columns[5]
    
    if data_col and idx_col:
        df['Total_KB'] = df[data_col] + df[idx_col]
    else:
        df['Total_KB'] = df[size_col]
        
    # Convert KB to GB (KB / 1024 / 1024)
    df['Size_GB'] = df['Total_KB'] / (1024 * 1024)
    
    # Group by company name
    grouped = df.groupby('Company Name')['Size_GB'].sum()
    return grouped.to_dict()

def run_analytics_and_forecast(input_dir, output_html, years_to_predict=3):
    files = scan_input_folder(input_dir)
    if len(files) < 2:
        return "ERROR: Minimum 2 files required in input_excel folder to calculate delta."
    
    all_companies = set()
    history_records = []
    
    for f in files:
        sizes = calculate_sizes(f)
        for comp, size in sizes.items():
            all_companies.add(comp)
            history_records.append({'date': f['date'], 'company': comp, 'size_gb': size})
            
    df_history = pd.DataFrame(history_records)
    
    # Create a pivot matrix (dates x companies), filling new companies with 0 for past dates
    df_pivot = df_history.pivot(index='date', columns='company', values='size_gb').fillna(0)
    
    # Calculate delta between the last two available files
    last_date = df_pivot.index[-1]
    prev_date = df_pivot.index[-2]
    days_delta = (last_date - prev_date).days
    
    delta_text = f"<h3>Database Size Changes (Last {days_delta} Days Period):</h3><ul>"
    for comp in df_pivot.columns:
        d = df_pivot.loc[last_date, comp] - df_pivot.loc[prev_date, comp]
        delta_text += f"<li><b>{comp}:</b> {'+' if d>=0 else ''}{d:.4f} GB</li>"
    total_d = df_pivot.loc[last_date].sum() - df_pivot.loc[prev_date].sum()
    delta_text += f"<li><b>TOTAL DATABASE SIZE:</b> {'+' if total_d>=0 else ''}{total_d:.4f} GB</li></ul>"

    # Linear Regression Forecasting
    start_date = df_pivot.index.min()
    X_train = (df_pivot.index - start_date).days.values.reshape(-1, 1)
    
    # Generate future dates (monthly intervals for N years ahead)
    future_dates = pd.date_range(start=last_date, periods=years_to_predict * 12, freq='ME')
    X_future = (future_dates - start_date).days.values.reshape(-1, 1)
    
    df_pred = pd.DataFrame(index=future_dates)
    
    for comp in df_pivot.columns:
        y_train = df_pivot[comp].values
        # If company is brand new (less than 2 data points), freeze its size to prevent model failure
        if np.count_nonzero(y_train) < 2:
            df_pred[comp] = y_train[-1]
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_future)
            df_pred[comp] = np.clip(preds, 0, None) # Storage size cannot be negative
            
    # Build Stacked Line Chart using Plotly
    fig = go.Figure()
    
    # Add historical data (solid lines with markers)
    for comp in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot.index, y=df_pivot[comp],
            mode='lines+markers', name=f"{comp} (History)",
            stackgroup='one'
        ))
        
    # Add prediction data (dashed lines)
    for comp in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_pred.index, y=df_pred[comp],
            mode='lines', name=f"{comp} (Forecast)",
            stackgroup='one',
            line=dict(dash='dash')
        ))
        
    fig.update_layout(
        title="Database Storage Capacity Forecast (Stacked)",
        xaxis_title="Date",
        yaxis_title="Storage Size (GB)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Generate HTML Report structure
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Database Monitor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="card shadow border-0 p-4 mb-4">
                <h1 class="text-primary">AI Agent Report: Storage Capacity Analysis</h1>
                <p class="text-muted">Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <hr>
                {delta_text}
            </div>
            <div class="card shadow border-0 p-4 mb-5">
                {graph_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    return "SUCCESS"

if __name__ == "__main__":
    res = run_analytics_and_forecast('../input_excel', '../docs/index.html')
    print(res)