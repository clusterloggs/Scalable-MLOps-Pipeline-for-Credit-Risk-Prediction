# Import necessary libraries
import dash  # Dash framework for building web applications
from dash import dcc, html, Input, Output, State  # Dash core components for interactivity and layout
import plotly.express as px  # High-level interface for creating visualizations
import plotly.graph_objects as go  # Low-level interface for advanced chart customization
import pandas as pd  # Library for data manipulation and analysis
import numpy as np  # Library for numerical computations
from sklearn.ensemble import RandomForestClassifier  # Machine learning model for classification
import dash_bootstrap_components as dbc  # Bootstrap components for styling Dash apps

# Load the dataset for machine learning
df = pd.read_csv(r"C:\Sachin\Document\credit_risk_dataset.csv")  # Load the dataset from a CSV file

# Define a dictionary to map old column names to new, more intuitive names
columns_to_rename = {
    'person_age': 'age',
    'person_home_ownership': 'home_status',
    'person_income': 'income',
    'person_emp_length': 'emp_years',
    'loan_amnt': 'loan_amount',
    'cb_person_default_on_file': 'credit_default',
    'cb_person_cred_hist_length': 'credit_history'
}
df.rename(columns=columns_to_rename, inplace=True)  # Rename columns in the DataFrame

# Impute missing values for numerical columns using the median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Identify numerical columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())  # Fill missing values with the median

# Impute missing values for categorical columns using the mode
categorical_cols = df.select_dtypes(include=['object']).columns  # Identify categorical columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])  # Fill missing values with the mode

# Map 'Y' to 1 and 'N' to 0 in the 'credit_default' column
df['credit_default'] = df['credit_default'].map({'Y': 1, 'N': 0})  # Convert categorical values to binary (1 or 0)

# One-hot encoding for categorical columns
categorical_columns = ['home_status', 'loan_intent', 'loan_grade']  # List of categorical columns to encode
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)  # Convert categorical variables to dummy variables

# Train the Random Forest model
features = ['loan_int_rate', 'loan_amount', 'credit_history']  # Features for the model
target = 'credit_default'  # Target variable for prediction
X = df[features]  # Feature matrix (independent variables)
y = df[target]  # Target vector (dependent variable)
rf_model = RandomForestClassifier(random_state=43)  # Initialize the Random Forest model with a fixed random seed
rf_model.fit(X, y)  # Train the model using the training data

# Load the dataset for charts
df_2 = pd.read_csv(r"C:\Sachin\Document\credit_risk_dataset.csv")  # Load the dataset again for visualization purposes

# Preprocessing for charts
df_2.rename(columns=columns_to_rename, inplace=True)  # Rename columns for consistency with the first dataset

# Initialize the Dash app with a Bootstrap theme (LUX in this case)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])  # Create a Dash app with the LUX Bootstrap theme

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
                    min=df_2['loan_int_rate'].min(),  # Minimum value of the slider
                    max=df_2['loan_int_rate'].max(),  # Maximum value of the slider
                    step=0.1,  # Step size for the slider
                    value=[df_2['loan_int_rate'].min(), df_2['loan_int_rate'].max()],  # Default value of the slider
                    marks={i: f'{i:.1f}' for i in np.linspace(df_2['loan_int_rate'].min(), df_2['loan_int_rate'].max(), 10)}  # Marks on the slider
                ),
                html.Label("Home Status"),  # Label for the home status dropdown
                dcc.Dropdown(  # Dropdown for selecting home status
                    id='home-status-dropdown',  # Unique ID for the dropdown
                    options=[{'label': 'All', 'value': 'All'}] +  # Add "All" option
                            [{'label': status, 'value': status} for status in df_2['home_status'].unique()],  # Options for the dropdown
                    value=['All'],  # Default value of the dropdown
                    multi=True  # Allow multiple selections
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
                    dbc.Row([  # Row for loan interest rate input
                        dbc.Col(html.Label("Loan Interest Rate (%)"), width=4),  # Label for the input
                        dbc.Col(dcc.Input(id='loan-int-rate-input', type='number', min=0, step=0.01), width=8)  # Input field
                    ], style={'margin-bottom': '10px'}),  # Add margin to the row
                    
                    dbc.Row([  # Row for loan amount input
                        dbc.Col(html.Label("Loan Amount (£)"), width=4),  # Label for the input
                        dbc.Col(dcc.Input(id='loan-amount-input', type='number', min=0), width=8)  # Input field
                    ], style={'margin-bottom': '10px'}),  # Add margin to the row
                    
                    dbc.Row([  # Row for credit history input
                        dbc.Col(html.Label("Credit History (years)"), width=4),  # Label for the input
                        dbc.Col(dcc.Input(id='credit-history-input', type='number', min=0, max=30), width=8)  # Input field
                    ], style={'margin-bottom': '10px'}),  # Add margin to the row
                    
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
    filtered_df = df_2[
        (df_2['loan_int_rate'] >= interest_rate_range[0]) & 
        (df_2['loan_int_rate'] <= interest_rate_range[1])
    ]
    
    # Filter based on home status
    if 'All' not in home_status:  # If "All" is not selected, filter by selected home statuses
        filtered_df = filtered_df[filtered_df['home_status'].isin(home_status)]
    
    # Donut Chart: Distribution of Defaulters
    intent_counts = filtered_df['credit_default'].value_counts()  # Count the occurrences of each credit default status
    donut_fig = go.Figure(data=[go.Pie(  # Create a donut chart
        labels=intent_counts.index,  # Labels for the chart
        values=intent_counts.values,  # Values for the chart
        hole=0.4,  # Size of the hole in the middle (creates a donut shape)
        marker_colors=px.colors.qualitative.Set3  # Use the Set3 color scheme
    )])
    donut_fig.update_layout(  # Update the layout of the chart
        title_text="Distribution of Defaulters",  # Title of the chart
        showlegend=True  # Show the legend
    )
    
    # Stacked Bar Chart: Loan Intent Distribution by Home Status
    grouped_df = filtered_df.groupby(['loan_grade', 'credit_default']).size().unstack(fill_value=0)  # Group data by loan grade and credit default status
    percent = grouped_df.div(grouped_df.sum(axis=1), axis=0) * 100  # Calculate percentages

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
        title_text="Loan Intent Distribution by Default Status",  # Title of the chart
        xaxis_title="Loan Intent",  # X-axis label
        yaxis_title="Percentage",  # Y-axis label
        barmode='stack',  # Stacked bar chart
        legend_title="Default Status"  # Legend title
    )

    return donut_fig, stacked_bar_fig  # Return both figures

# Callback for loan default prediction
@app.callback(
    Output('prediction-output', 'children'),  # Output: Prediction result
    [Input('predict-button', 'n_clicks')],  # Input: Number of clicks on the predict button
    [State('loan-int-rate-input', 'value'),  # State: Loan interest rate input
     State('loan-amount-input', 'value'),  # State: Loan amount input
     State('credit-history-input', 'value')]  # State: Credit history input
)
def predict_default(n_clicks, loan_int_rate, loan_amount, credit_history):
    if n_clicks > 0:  # Check if the predict button has been clicked
        try:
            # Validate inputs
            if loan_int_rate is None or loan_amount is None or credit_history is None:
                return "Please fill in all fields."  # Return error message if any input is missing
            
            if loan_int_rate < 0 or loan_amount < 0 or credit_history < 0 or credit_history > 30:
                return "Invalid input values. Please check your inputs."  # Return error message if inputs are invalid
            
            # Make prediction
            prediction = rf_model.predict([[loan_int_rate, loan_amount, credit_history]])  # Predict using the trained model
            prediction_proba = rf_model.predict_proba([[loan_int_rate, loan_amount, credit_history]])  # Get prediction probabilities
            
            # Format output
            result = "Yes" if prediction[0] == 1 else "No"  # Convert prediction to "Yes" or "No"
            probability = f"{prediction_proba[0][1]:.2%}"  # Format probability as a percentage
            
            return html.Div([  # Return the prediction result
                html.P(f"Will the borrower default on the loan?: {result}"),  # Display prediction result
                html.P(f"Probability of default: {probability}")  # Display probability of default
            ])
        
        except Exception as e:
            return f"An error occurred: {str(e)}"  # Return error message if an exception occurs
    
    return ""  # Return empty string if the predict button has not been clicked

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)  # Run the Dash app in non-debug mode