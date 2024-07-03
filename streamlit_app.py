#################
TEST_PREDS_PATH = "experiments/test_preds"
RESERVATION_PATH = "dynamic_reservations_feat_new.pq"
#################

import streamlit as st
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import yaml
import pandas as pd
import os 
import glob

# Define the YAML configuration for authentication
config = {
    'credentials': {
        'usernames': {
            'hc': {
                'name': 'User One',
                'password': 'hc_cloud'
            },
            'sanskar': {
                'name': 'User Two',
                'password': 'sanskar'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'some_signature_key',
        'name': 'some_cookie_name'
    },
    'preauthorized': {
        'emails': ['email@domain.com']
    }
}

# Save the configuration to a file
with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

# Load the configuration file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Initialize the authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Display the login form
auth_status = authenticator.login('main', fields = {'Form name': 'Welcome'})

if auth_status[1]:
    st.write(f'Welcome *{auth_status[0]}*')

    # Your Plotly plots go here
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines', name='lines'))
    st.plotly_chart(fig)
    

    data = pd.read_parquet(RESERVATION_PATH)
    data = data.query(f'lead_in <=120 & lead_in >=3')
    test_start_date = "2024-01-01"
    data = data[data["stay_date"] >= test_start_date]
    data["stay_day_of_week"] = data["stay_date"].dt.day_of_week




    model_names = glob.glob(f"{TEST_PREDS_PATH}/*.csv")
    model_names = [os.path.basename(m).replace(".csv", "") for m in model_names]

    for model in model_names:
        preds = pd.read_csv(f"experiments/test_preds/{model}.csv")
        assert len(preds) == len(data)
        data[model] = preds["preds"].values

        data[f'{model}_error'] = data[model] - data["pickup_3"]
        data[f'abs_error_{model}'] = abs(data[f'{model}_error'])
        data[f'error_cum_abs_{model}'] = data.groupby("stay_date")[f'abs_error_{model}'].cumsum()
        data[f'error_cum_{model}'] = data.groupby("stay_date")[f'{model}_error'].cumsum()

        data[f'{model}_cumulative_3'] = data[model] + data["cumulative_reservations"]

        

    data = data.query(f'lead_in <=30 & lead_in >=3')

    fig = go.Figure()


    dropdown_buttons = []
    number_of_models = len(model_names) + 1

    for lead_in in range(4, 28):
        lead_in_data = data[data["lead_in"] == lead_in]
        
        
        actual_means = lead_in_data.groupby("stay_day_of_week")["pickup_3"].mean()
        fig.add_trace(go.Bar(y=actual_means, name="Actual", marker=dict(line=dict(color='black', width=2)), visible=(lead_in == 4)))
        
        
        for model_name in model_names:
            predicted_means = lead_in_data.groupby("stay_day_of_week")[model_name].mean()
            fig.add_trace(go.Bar(y=predicted_means, name=model_name, visible=(lead_in == 4)))
        
        visibility_array = [False] * number_of_models * (28 - 4)
        start_index = number_of_models * (lead_in - 4)
        for i in range(number_of_models):
            visibility_array[start_index + i] = True

        dropdown_buttons.append(
            dict(
                label=f'Lead_in {lead_in}',
                method='update',
                args=[{'visible': visibility_array,
                    'title': f'Lead_in {lead_in}'}
                    ]
            )
        )
        
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=1.05,  
                xanchor='left',
                y=1.15,  
                yanchor='top'
            )
        ],
        title='Lead_in Analysis per Model',
        title_x=0.4,
        barmode='group'  
    )

    fig.update_xaxes(title_text="Day of Week")
    fig.update_yaxes(title_text="3 days Look Ahead Individual Reservations")

    st.plotly_chart(fig)

    stay_range = sorted([str(date.date()) for date in data['stay_date'].unique()])

    fig = go.Figure()

    for stay_date in stay_range:
        filtered_data = data[data["stay_date"] == stay_date]
        
        fig.add_trace(go.Scatter(
        x=filtered_data["lead_in"],
        y=filtered_data["pickup_3"],
        mode="markers+lines",
        name="Actual",
        marker=dict(color="red", symbol="circle", size=3),  
        line=dict(color="black", width=2, dash='dash'),  
        visible=(stay_date == stay_range[0])  
    ))
        
        for model in model_names:
            fig.add_trace(go.Scatter(
                x=filtered_data["lead_in"],
                y=filtered_data[model],
                mode="markers+lines",
                name=f"{model} , MAE: {filtered_data[f'abs_error_{model}'].mean():.2f}",
                marker=dict(size=2),
                visible=(stay_date == stay_range[0])  
            ))


    buttons = []
    for stay_date in stay_range:
        visible = [date == stay_date for date in stay_range for _ in (range(1 + len(model_names)))]
        button = dict(
            label=str(stay_date),
            method="update",
            args=[{"visible": visible},
                {"title": f"Actual and Predicted Pickup, Stay Date: {stay_date}"}])
        buttons.append(button)

    fig.update_layout(
        title=f"Actual and Predicted Pickup, Stay Date: {stay_range[0]}",
        title_x = 0.45,
        xaxis_title="Lead In",
        yaxis_title="3D Look Ahead Reservations",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    st.plotly_chart(fig)

    stay_range = sorted([str(date.date()) for date in data['stay_date'].unique()])

    fig = go.Figure()

    for stay_date in stay_range:
        filtered_data = data[data["stay_date"] == stay_date]
        
        fig.add_trace(go.Scatter(
        x=filtered_data["lead_in"],
        y=filtered_data["cumulative_reservations_3"],
        mode="markers+lines",
        name="Actual Reservations",
        marker=dict(color="red", symbol="circle", size=3),  
        line=dict(color="black", width=2, dash='dash'),  
        visible=(stay_date == stay_range[0])  
    ))
        
        for model in model_names:
            diff = (filtered_data[f'{model}_cumulative_3'] - filtered_data["cumulative_reservations_3"]).mean()
            fig.add_trace(go.Scatter(
                x=filtered_data["lead_in"],
                y=filtered_data[f'{model}_cumulative_3'],
                mode="markers+lines",
                name=f"{model} , Cumulative Error: {diff:.2f}",
                marker=dict(size=2),
                visible=(stay_date == stay_range[0])  
            ))


    buttons = []
    for stay_date in stay_range:
        visible = [date == stay_date for date in stay_range for _ in (range(1 + len(model_names)))]
        button = dict(
            label=str(stay_date),
            method="update",
            args=[{"visible": visible},
                {"title": f"Actual and Predicted Cumulative Reservations, Stay Date: {stay_date}"}])
        buttons.append(button)

    fig.update_layout(
        title=f"Actual and Predicted Cumulative Reservations, Stay Date: {stay_range[0]}",
        title_x = 0.45,
        xaxis_title="Lead In",
        yaxis_title="Cumulative Reservations",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    fig.update_xaxes(autorange="reversed")

    st.plotly_chart(fig)

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=[0, 32],  
        y=[0, 0],  
        mode="lines",
        name="y=0 (Zero Error Line)",
        line=dict(color="black", width=2, dash='dash'),
        visible=True  
    ))


    for stay_date in stay_range:
        filtered_data = data[data["stay_date"] == stay_date]
        
        for model in model_names:
            mean_error = filtered_data[f'{model}_error'].mean()
            fig.add_trace(go.Scatter(
                x=filtered_data["lead_in"],
                y=filtered_data[f'{model}_error'],
                mode="markers+lines",
                name=f"{model}, Mean Error : {mean_error:.2f}",
                marker=dict(size=2),
                visible=(stay_date == stay_range[0])  #
            ))

    buttons = []
    for i, stay_date in enumerate(stay_range):
        
        visible = [True] + [False] * len(model_names) * len(stay_range)
        start = 1 + i * len(model_names)  # Adjust start point for visibility toggling
        visible[start:start + len(model_names)] = [True] * len(model_names)
        button = dict(
            label=str(stay_date),
            method="update",
            args=[{"visible": visible},
                {"title": f"Error Analysis, Stay Date: {stay_date}"}])
        buttons.append(button)


    fig.update_layout(
        title=f"Error Analysis, Stay Date: {stay_range[0]}",
        title_x=0.45,
        xaxis_title="Lead In",
        yaxis_title="Error",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    st.plotly_chart(fig)


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 32],  
        y=[0, 0],  
        mode="lines",
        name="y=0 (Zero Error Line)",
        line=dict(color="black", width=2, dash='dash'),
        visible=True  
    ))

    for stay_date in stay_range:
        filtered_data = data[data["stay_date"] == stay_date]
        
        for model in model_names:
            error = filtered_data[f'error_cum_{model}'].mean()
            fig.add_trace(go.Scatter(
                x=filtered_data["lead_in"],
                y=filtered_data[f'error_cum_{model}'],
                mode="markers+lines",
                name=f"{model}, Cumulative Error : {error:.2f}",
                marker=dict(size=2),
                visible=(stay_date == stay_range[0])  
            ))

    buttons = []
    for i, stay_date in enumerate(stay_range):
        
        visible = [True] + [False] * len(model_names) * len(stay_range)
        start = 1 + i * len(model_names)  
        visible[start:start + len(model_names)] = [True] * len(model_names)
        button = dict(
            label=str(stay_date),
            method="update",
            args=[{"visible": visible},
                {"title": f"Cumulative Error Analysis, Stay Date: {stay_date}"}])
        buttons.append(button)


    fig.update_layout(
        title=f"Cumulative Error Analysis, Stay Date: {stay_range[0]}",
        title_x=0.45,
        xaxis_title="Lead In",
        yaxis_title="Cumulative Error",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )


    st.plotly_chart(fig)

    fig = go.Figure()

    for stay_date in stay_range:
        filtered_data = data[data["stay_date"] == stay_date]
        
        
        for model in model_names:
            abs_cum_error = filtered_data[f'error_cum_abs_{model}'].mean()
            fig.add_trace(go.Scatter(
                x=filtered_data["lead_in"],
                y=filtered_data[f'error_cum_abs_{model}'],
                mode="markers+lines",
                name=f"{model}, Absolute Cumulative Error : {abs_cum_error:.2f}",
                marker=dict(size=2),
                visible=(stay_date == stay_range[0])  
            ))

    buttons = []
    for i, stay_date in enumerate(stay_range):
        
        visible = [False] * len(model_names) * len(stay_range)
        start = 1 + i * len(model_names)  
        visible[start:start + len(model_names)] = [True] * len(model_names)
        button = dict(
            label=str(stay_date),
            method="update",
            args=[{"visible": visible},
                {"title": f"Cumulative Error Analysis Absolute, Stay Date: {stay_date}"}])
        buttons.append(button)


    fig.update_layout(
        title=f"Cumulative Error Analysis ( Absolute ), Stay Date: {stay_range[0]}",
        title_x=0.45,
        xaxis_title="Lead In",
        yaxis_title="Absolute Cumulative Error",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.03,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )

    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig)

    authenticator.logout('Logout', 'sidebar')


elif auth_status[1] == False:
    st.error('Username/password is incorrect')

elif auth_status[1] == None:
    st.warning('Please enter your username and password')
    