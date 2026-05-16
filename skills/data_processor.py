import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from anthropic import Anthropic

def extract_date_from_filename(filename):
    match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    if match:
        return datetime.strptime(match.group(0), "%Y-%m-%d")
    return None

def scan_input_folder(folder_path):
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
    if file_info['path'].endswith('.csv'):
        df = pd.read_csv(file_info['path'])
    else:
        df = pd.read_excel(file_info['path'])
    
    df['Company Name'] = df['Company Name'].fillna('System / Shared')
    if hasattr(df['Company Name'], 'str'):
        df['Company Name'] = df['Company Name'].str.strip()
    
    data_col = 'Data Size (KB)' if 'Data Size (KB)' in df.columns else None
    idx_col = 'Index Size (KB)' if 'Index Size (KB)' in df.columns else None
    size_col = 'Size (KB)' if 'Size (KB)' in df.columns else df.columns[5]
    
    if data_col and idx_col:
        df['Total_KB'] = df[data_col] + df[idx_col]
    else:
        df['Total_KB'] = df[size_col]
        
    df['Size_GB'] = df['Total_KB'] / (1024 * 1024)
    return df.groupby('Company Name')['Size_GB'].sum().to_dict()

def load_company_profiles(skills_dir):
    profile_path = os.path.join(skills_dir, 'company_profiles.json')
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_ai_analysis(delta_metrics, forecast_summary, profiles):
    """Sends combined math data and company profiles to Anthropic Claude"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "<div class='alert alert-warning'>AI Analysis skipped: ANTHROPIC_API_KEY environment variable not found.</div>"
    
    try:
        client = Anthropic(api_key=api_key)
        
        # Preparing the context string for the AI prompt
        prompt_data = f"--- MATHEMATICAL DELTA METRICS ---\n{delta_metrics}\n\n"
        prompt_data += f"--- 3-YEAR FORECAST LINEAR TRENDS ---\n{forecast_summary}\n\n"
        prompt_data += f"--- COMPANY BUSINESS PROFILES & CONTEXT ---\n{json.dumps(profiles, indent=2)}\n"
        
        message = client.messages.create(
            model="claude-haiku-4-5-20251001", # Standard high-performing engine
            max_tokens=1500,
            temperature=0.3,
            system="You are an expert Database Administrator and Business Data Analyst. Analyze the database storage growth against the provided company business profiles (shipment volumes, customer growth plans). Provide an executive summary explaining the correlation between business activity and database size changes, highlight anomalies, and issue capacity planning recommendations. Use clean HTML tags (like <h4>, <p>, <ul>, <li>, <strong>) for formatting. Do not output markdown or complete <html> wrappers.",
            messages=[
                {"role": "user", "content": f"Analyze the following data and generate the report content:\n\n{prompt_data}"}
            ]
        )
        return f"<div class='mt-3'>{message.content[0].text}</div>"
    except Exception as e:
        return f"<div class='alert alert-danger'>AI Generation Error: {str(e)}</div>"

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
    df_pivot = df_history.pivot(index='date', columns='company', values='size_gb').fillna(0)
    
    last_date = df_pivot.index[-1]
    prev_date = df_pivot.index[-2]
    days_delta = (last_date - prev_date).days
    
    # Building mathematical metrics for report and AI prompt
    delta_text = f"<h3>Database Size Changes (Last {days_delta} Days Period):</h3><ul>"
    raw_delta_log = ""
    for comp in df_pivot.columns:
        d = df_pivot.loc[last_date, comp] - df_pivot.loc[prev_date, comp]
        delta_text += f"<li><b>{comp}:</b> {'+' if d>=0 else ''}{d:.4f} GB</li>"
        raw_delta_log += f"- {comp}: Last size {df_pivot.loc[last_date, comp]:.4f} GB, Delta change {'+' if d>=0 else ''}{d:.4f} GB\n"
    total_d = df_pivot.loc[last_date].sum() - df_pivot.loc[prev_date].sum()
    delta_text += f"<li><b>TOTAL DATABASE SIZE:</b> {'+' if total_d>=0 else ''}{total_d:.4f} GB</li></ul>"
    raw_delta_log += f"- OVERALL TOTAL: Delta {'+' if total_d>=0 else ''}{total_d:.4f} GB\n"

    # Forecasting
    start_date = df_pivot.index.min()
    X_train = (df_pivot.index - start_date).days.values.reshape(-1, 1)
    future_dates = pd.date_range(start=last_date, periods=years_to_predict * 12, freq='ME')
    X_future = (future_dates - start_date).days.values.reshape(-1, 1)
    df_pred = pd.DataFrame(index=future_dates)
    
    raw_forecast_log = ""
    for comp in df_pivot.columns:
        y_train = df_pivot[comp].values
        if np.count_nonzero(y_train) < 2:
            df_pred[comp] = y_train[-1]
            raw_forecast_log += f"- {comp}: Static prediction at {y_train[-1]:.4f} GB\n"
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_future)
            df_pred[comp] = np.clip(preds, 0, None)
            raw_forecast_log += f"- {comp}: Calculated 3-year end size will reach {df_pred[comp].iloc[-1]:.4f} GB\n"
            
    # Build Stacked Line Chart
    fig = go.Figure()
    for comp in df_pivot.columns:
        fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[comp], mode='lines+markers', name=f"{comp} (History)", stackgroup='one'))
    for comp in df_pred.columns:
        fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred[comp], mode='lines', name=f"{comp} (Forecast)", stackgroup='one', line=dict(dash='dash')))
        
    fig.update_layout(title="Database Storage Capacity Forecast (Stacked)", xaxis_title="Date", yaxis_title="Storage Size (GB)", template="plotly_white", hovermode="x unified")
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # AI EXECUTION BLOCK
    skills_dir = os.path.dirname(os.path.abspath(__file__))
    profiles = load_company_profiles(skills_dir)
    print("[AI-Skill] Requesting intelligence analysis from Claude...")
    ai_insights = get_ai_analysis(raw_delta_log, raw_forecast_log, profiles)
    
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
            
            <div class="card shadow border-0 p-4 mb-4 bg-white">
                <h3 class="text-info">🤖 AI Executive Insights & Business Correlation</h3>
                <hr>
                {ai_insights}
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