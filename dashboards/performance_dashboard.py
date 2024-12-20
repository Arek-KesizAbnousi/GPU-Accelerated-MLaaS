# dashboards/performance_dashboard.py

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# Load training metrics
df = pd.read_csv('training_metrics.csv')

app.layout = html.Div([
    html.H1('Model Training Performance'),
    dcc.Graph(
        figure=px.line(df, y=['accuracy', 'val_accuracy'], title='Model Accuracy', labels={'index': 'Epoch'})
    ),
    dcc.Graph(
        figure=px.line(df, y=['loss', 'val_loss'], title='Model Loss', labels={'index': 'Epoch'})
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
