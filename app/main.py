
# Import necessary libraries
import dash  # Dash framework for building web applications
from dash import dcc, html, Input, Output, State  # Dash core components for interactivity and layout
import plotly.express as px  # High-level interface for creating visualizations
import plotly.graph_objects as go  # Low-level interface for advanced chart customization
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical computations
import joblib  # For loading the trained model
import os  # For path operations
import dash_bootstrap_components as dbc  # Bootstrap components for styling Dash apps


# --- Model Loading ---
# Load the trained model pipeline once when the application starts.
MODEL_PATH = 'models/model.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"Error: Model file not found at {MODEL_PATH}")

# --- Data Loading for UI & Visualizations ---
# Load the dataset to populate dropdowns and power visualizations.
# Assumes the app is run from the project's root directory.
try:
    df_viz = pd.read_parquet("data/df_clean.parquet")
except FileNotFoundError:
    print("Warning: data/df_clean.parquet not found. Visualizations and dropdowns may not work correctly.")
    # Create an empty dataframe with expected columns to prevent app from crashing
    df_viz = pd.DataFrame({col: [] for col in [
        'income', 'emp_years', 'loan_amount', 'loan_int_rate', 'loan_percent_income',
        'credit_history', 'home_status', 'loan_intent', 'loan_grade', 'default'
    ]})

# Get unique values for dropdowns from the dataset
home_status_options = [{'label': i, 'value': i} for i in df_viz['home_status'].unique()]
loan_intent_options = [{'label': i, 'value': i} for i in df_viz['loan_intent'].unique()]
loan_grade_options = [{'label': i, 'value': i} for i in sorted(df_viz['loan_grade'].unique())]


# Initialize the Dash app with a Bootstrap theme (LUX in this case)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])  # Create a Dash app with the LUX Bootstrap theme
server = app.server # Expose server for WSGI


# Layout of the app
app.layout = html.Div([  # Define the layout of the app using HTML-like components
    dcc.Tabs([  # Create tabs for the app
        # Tab 1: Credit Risk Analysis Dashboard
        dcc.Tab(label='Credit Risk Analysis', children=[  # First tab for credit risk analysis
            html.H1("Credit Risk Analysis Dashboard", style={'textAlign': 'center'}),  # Title for the dashboard
           
            # Filters for the dashboard
            html.Div([  # Container for filters
                html.Label("Interest Rate Range"),  # Label for the interest rate slider
                dcc.RangeSlider(  # Range slider for selecting interest rate range
                    id='interest-rate-slider',  # Unique ID for the slider
                    min=df_viz['loan_int_rate'].min(),  # Minimum value of the slider
                    max=df_viz['loan_int_rate'].max(),  # Maximum value of the slider
                    step=0.1,  # Step size for the slider
                    value=[df_viz['loan_int_rate'].min(), df_viz['loan_int_rate'].max()],  # Default value of the slider
                    marks={str(round(i, 1)): f'{i:.1f}' for i in np.linspace(df_viz['loan_int_rate'].min(), df_viz['loan_int_rate'].max(), 10)}  # Marks on the slider
                ),
                html.Label("Home Status"),  # Label for the home status dropdown
                dcc.Dropdown(  # Dropdown for selecting home status
                    id='home-status-dropdown',  # Unique ID for the dropdown
                    options=[{'label': 'All', 'value': 'All'}] +  # Add "All" option
                            [{'label': status, 'value': status} for status in df_viz['home_status'].unique()],  # Options for the dropdown
                    value=['All'],  # Default value of the dropdown
                    multi=True,  # Allow multiple selections
                )
            ], style={'margin': '20px'}),  # Add margin to the filter container
           
            # Charts (placed side by side)
            html.Div([  # Container for charts
                # Donut Chart
                html.Div(
                    dcc.Graph(id='donut-chart'),  # Donut chart for visualizing data
                    style={'width': '48%', 'display': 'inline-block'}  # Style for the donut chart container
                ),
                # Stacked Bar Chart
                html.Div(
                    dcc.Graph(id='stacked-bar-chart'),  # Stacked bar chart for visualizing data
                    style={'width': '48%', 'display': 'inline-block'}  # Style for the stacked bar chart container
                )
            ], style={'display': 'flex', 'justify-content': 'space-between'})  # Flexbox layout for charts
        ]),
       
        # Tab 2: Loan Default Prediction
        dcc.Tab(label='Loan Default Prediction', children=[  # Second tab for loan default prediction
            html.H1("Loan Default Prediction", style={'textAlign': 'center'}),  # Title for the prediction tab
           
            # Input fields for prediction
            dbc.Card(  # Card container for input fields
                dbc.CardBody([  # Body of the card
                    # Numerical Inputs
                    dbc.Row([
                        dbc.Col(html.Label("Annual Income"), width=4),
                        dbc.Col(dcc.Input(id='income-input', type='number', min=0, placeholder="e.g., 60000"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Employment Length (years)"), width=4),
                        dbc.Col(dcc.Input(id='emp-years-input', type='number', min=0, max=45, placeholder="e.g., 5"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Loan Amount"), width=4),
                        dbc.Col(dcc.Input(id='loan-amount-input', type='number', min=0, placeholder="e.g., 10000"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Loan Interest Rate (%)"), width=4),
                        dbc.Col(dcc.Input(id='loan-int-rate-input', type='number', min=0, step=0.01, placeholder="e.g., 7.5"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Loan as % of Income"), width=4),
                        dbc.Col(dcc.Input(id='loan-percent-income-input', type='number', min=0, max=1, step=0.01, placeholder="e.g., 0.17"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Credit History (years)"), width=4),
                        dbc.Col(dcc.Input(id='credit-history-input', type='number', min=0, max=30, placeholder="e.g., 6"), width=8)
                    ], style={'margin-bottom': '10px'}),
                    # Categorical Inputs
                    dbc.Row([
                        dbc.Col(html.Label("Home Ownership"), width=4),
                        dbc.Col(dcc.Dropdown(id='home-status-input', options=home_status_options), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Loan Intent"), width=4),
                        dbc.Col(dcc.Dropdown(id='loan-intent-input', options=loan_intent_options), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([
                        dbc.Col(html.Label("Loan Grade"), width=4),
                        dbc.Col(dcc.Dropdown(id='loan-grade-input', options=loan_grade_options), width=8)
                    ], style={'margin-bottom': '10px'}),
                    dbc.Row([  # Row for the predict button
                        dbc.Col(dbc.Button('Predict', id='predict-button', n_clicks=0, color='primary'), width=12)  # Predict button
                    ])
                ]),
                style={'margin': '20px', 'padding': '20px'}  # Style for the card
            ),
           
            # Display prediction results
            dbc.Card(  # Card container for prediction results
                dbc.CardBody([  # Body of the card
                    html.H3("Prediction Result", style={'textAlign': 'center'}),  # Title for the prediction result
                    html.Div(id='prediction-output')  # Container for displaying prediction results
                ]),
                style={'margin': '20px', 'padding': '20px'}  # Style for the card
            )
        ])
    ])
])


# Callback for updating charts based on filters
@app.callback(
    [Output('donut-chart', 'figure'),  # Output: Donut chart figure
     Output('stacked-bar-chart', 'figure')],  # Output: Stacked bar chart figure
    [Input('interest-rate-slider', 'value'),  # Input: Interest rate range
     Input('home-status-dropdown', 'value')]  # Input: Home status selection
)
def update_charts(interest_rate_range, home_status):
    # Filter the dataset based on the selected interest rate range
    filtered_df = df_viz[
        (df_viz['loan_int_rate'] >= interest_rate_range[0]) &
        (df_viz['loan_int_rate'] <= interest_rate_range[1])
    ]
   
    # Filter based on home status
    if 'All' not in home_status:  # If "All" is not selected, filter by selected home statuses
        filtered_df = filtered_df[filtered_df['home_status'].isin(home_status)]
   
    # Donut Chart: Distribution of Defaulters
    intent_counts = filtered_df['default'].value_counts()  # Count the occurrences of each credit default status
    donut_labels = ['No Default' if i == 0 else 'Default' for i in intent_counts.index] # Dynamic labels
    donut_fig = go.Figure(data=[go.Pie(  # Create a donut chart
        labels=donut_labels,  # Use dynamic labels
        values=intent_counts.values,  # Values for the chart
        hole=0.4,  # Size of the hole in the middle (creates a donut shape)
        marker_colors=px.colors.qualitative.Set3  # Use the Set3 color scheme
    )])
    donut_fig.update_layout(  # Update the layout of the chart
        title_text="Distribution of Defaulters",  # Title of the chart
        showlegend=True  # Show the legend
    )
   
    # Stacked Bar Chart: Loan Intent Distribution by Home Status
    if not filtered_df.empty:
        grouped_df = filtered_df.groupby(['loan_grade', 'default']).size().unstack(fill_value=0)

        # Ensure both default (1) and no-default (0) columns exist to prevent errors
        if 0 not in grouped_df.columns:
            grouped_df[0] = 0
        if 1 not in grouped_df.columns:
            grouped_df[1] = 0
        
        # Ensure consistent column order and rename for the legend
        grouped_df = grouped_df[[0, 1]]
        grouped_df.columns = ['No Default', 'Default']
        percent = grouped_df.div(grouped_df.sum(axis=1).replace(0, 1), axis=0) * 100
    else:
        # Create an empty dataframe with correct columns if filter results in no data
        percent = pd.DataFrame(columns=['No Default', 'Default'])


    # Define the Set3 color scheme
    color_scheme = px.colors.qualitative.Set3


    # Create the stacked bar chart
    stacked_bar_fig = go.Figure()


    # Add traces for each credit default status
    for i, status in enumerate(percent.columns):
        stacked_bar_fig.add_trace(go.Bar(
            x=percent.index,  # X-axis: Loan grades
            y=percent[status],  # Y-axis: Percentage of each credit default status
            name=str(status),  # Name of the trace (credit default status)
            marker_color=color_scheme[i]  # Apply Set3 color scheme
        ))


    # Update layout
    stacked_bar_fig.update_layout(
        title_text="Loan Grade Distribution by Default Status",  # Title of the chart
        xaxis_title="Loan Grade",  # X-axis label
        yaxis_title="Percentage",  # Y-axis label
        barmode='stack',  # Stacked bar chart
        legend_title="Default Status"  # Legend title
    )


    return donut_fig, stacked_bar_fig  # Return both figures


# Callback for loan default prediction
@app.callback(
    Output('prediction-output', 'children'),  # Output: Prediction result
    [Input('predict-button', 'n_clicks')],  # Input: Number of clicks on the predict button
    [State('income-input', 'value'),
     State('emp-years-input', 'value'),
     State('loan-amount-input', 'value'),
     State('loan-int-rate-input', 'value'),
     State('loan-percent-income-input', 'value'),
     State('credit-history-input', 'value'),
     State('home-status-input', 'value'),
     State('loan-intent-input', 'value'),
     State('loan-grade-input', 'value')]
)
def predict_default(n_clicks, income, emp_years, loan_amount, loan_int_rate,
                    loan_percent_income, credit_history, home_status,
                    loan_intent, loan_grade):
    if n_clicks > 0:  # Check if the predict button has been clicked
        if model is None:
            return dbc.Alert("Model is not loaded. Cannot make a prediction.", color="danger")

        try:
            # Validate inputs
            all_inputs = [
                income, emp_years, loan_amount, loan_int_rate, loan_percent_income,
                credit_history, home_status, loan_intent, loan_grade
            ]
            if any(i is None for i in all_inputs):
                return dbc.Alert("Please fill in all fields to get a prediction.", color="warning")
           
            # Create a DataFrame with the correct feature names for the model pipeline
            feature_names = [
                'income', 'emp_years', 'loan_amount', 'loan_int_rate',
                'loan_percent_income', 'credit_history', 'home_status',
                'loan_intent', 'loan_grade'
            ]
            input_data = pd.DataFrame([all_inputs], columns=feature_names)

            # Ensure correct data types for numerical columns
            for col in ['income', 'emp_years', 'loan_amount', 'loan_int_rate', 'loan_percent_income', 'credit_history']:
                input_data[col] = pd.to_numeric(input_data[col])
           
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
           
            # Format output
            result = "Yes" if prediction[0] == 1 else "No"  # Convert prediction to "Yes" or "No"
            probability = f"{prediction_proba[0][1]:.2%}"  # Format probability as a percentage
           
            return html.Div([  # Return the prediction result
                html.P(f"Will the borrower default on the loan? {result}", className="lead"),
                html.P(f"Probability of default: {probability}", className="lead")
            ])
       
        except Exception as e:
            return dbc.Alert(f"An error occurred during prediction: {str(e)}", color="danger")
   
    return ""  # Return empty string if the predict button has not been clicked


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
