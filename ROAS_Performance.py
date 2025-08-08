import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Set Streamlit page configuration.
st.set_page_config(layout="wide")

# A function to load and preprocess data from the uploaded file.
@st.cache_data
def load_data(uploaded_file):
    """
    Loads and preprocesses the CSV data.
    """
    df = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in df.columns:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
    roas_columns = [col for col in df.columns if 'roas_d' in col]
    for col in roas_columns:
        # Check if column is already numeric, if not convert
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace('%', ''), errors='coerce') / 100
    
    df['max_roas'] = df[roas_columns].max(axis=1)
    
    # Extract day numbers for dynamic filter
    roas_days = sorted([int(col.replace('roas_d', '')) for col in roas_columns if col != 'roas_d0'])
    return df, roas_columns, roas_days

# Use session state to manage app state.
if 'show_analysis_page' not in st.session_state:
    st.session_state.show_analysis_page = False
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.roas_columns = None
    st.session_state.roas_days = None
    st.session_state.uploaded_filename = None

# Logic to automatically switch to analysis page if data is already in session state
if not st.session_state.show_analysis_page and st.session_state.df is not None:
    st.session_state.show_analysis_page = True
    st.rerun()

if not st.session_state.show_analysis_page:
    # This is the intro page with the file uploader.
    st.title("Welcome to the Campaign Performance Analyzer")
    st.write("This application helps you deeply analyze your marketing campaign data by providing insights into ROAS performance, cumulative growth and break-even projections.")
    st.write("To begin, please upload your CSV file below.")
    
    st.markdown("---")

    # The new link to download the report file, with app_token removed
    report_link = "https://suite.adjust.com/datascape/report?utc_offset=%2B00%3A00&reattributed=all&attribution_source=first&attribution_type=all&ad_spend_mode=network&date_period=-92d%3A-1d&cohort_maturity=mature&sandbox=false&channel_id__in=%22partner_257%22%2C%22partner_7%22%2C%22partner_34%22%2C%22partner_182%22%2C%22partner_100%22%2C%22partner_369%22%2C%22partner_56%22%2C%22partner_490%22%2C%22partner_2337%2C1678%22%2C%22partner_217%22&applovin_mode=probabilistic&ironsource_mode=ironsource&dimensions=app%2Cchannel%2Ccampaign_network&format_dates=false&full_data=true&include_attr_dependency=true&metrics=cost%2Cinstalls%2Cgross_profit%2Croas_d0%2Croas_d3%2Croas_d7%2Croas_d14%2Croas_d21%2Croas_d28%2Croas_d30%2Croas_d45%2Croas_d50&readable_names=false&sort=-cost&parent_report_id=213219&cost__gt__column=0&is_report_setup_open=true&table_view=pivot"
    st.markdown(f"**Don't have the report file?** You can download it directly from here: [**Adjust Report Link**]({report_link})")
    
    # File uploader on the main page
    uploaded_file = st.file_uploader("Upload your CSV file:", type=['csv'])

    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.df, st.session_state.roas_columns, st.session_state.roas_days = load_data(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            st.success("File uploaded successfully! You can now proceed to the analysis.")
            
            # Automatically transition to the analysis page after successful upload
            st.session_state.show_analysis_page = True
            st.rerun()
    
    if st.session_state.df is not None:
        if st.button("Clear File", use_container_width=True):
            st.session_state.df = None
            st.session_state.uploaded_filename = None
            st.session_state.show_analysis_page = False
            st.rerun()
    
else:
    # This is the main analysis page with tabs.
    st.header("Campaign Performance and Break-Even Analysis")
    st.write("Use the sidebar and tabs to filter and analyze your data.")
    
    # Sidebar for general analysis settings.
    st.sidebar.header("Analysis Settings")
    
    if st.session_state.df is not None:
        if st.sidebar.button("Clear Data"):
            st.session_state.df = None
            st.session_state.uploaded_filename = None
            st.session_state.show_analysis_page = False
            st.rerun()
            
        apps = st.session_state.df['app'].unique().tolist()
        selected_app = st.sidebar.selectbox("Select App:", apps)
        
        app_df = st.session_state.df[st.session_state.df['app'] == selected_app]
        networks = app_df['channel'].unique().tolist()
        selected_network = st.sidebar.selectbox("Select Network:", networks)

        roas_columns = st.session_state.roas_columns
        days_to_exclude = st.sidebar.number_input(
            "Exclude Last N Days from Analysis:",
            min_value=0,
            max_value=len(roas_columns) - 1,
            value=0,
            step=1,
            help="Enter the number of recent days to exclude from all analyses and charts due to unreliable data."
        )
        
        if days_to_exclude > 0:
            roas_columns_filtered = roas_columns[:-days_to_exclude]
            roas_days_filtered = st.session_state.roas_days[:-days_to_exclude]
        else:
            roas_columns_filtered = roas_columns
            roas_days_filtered = st.session_state.roas_days

        all_campaigns_in_network = app_df[app_df['channel'] == selected_network]['campaign_network'].unique().tolist()
        selected_campaigns = st.multiselect(
            "Select Campaigns:",
            all_campaigns_in_network,
            default=all_campaigns_in_network[:5]
        )
        
        tab1, tab2 = st.tabs(["Standard Analysis", "Scenario Analysis"])

        with tab1:
            st.header("Standard Break-Even Analysis")
            st.write("Analyze break-even for a single scenario based on historical growth.")
            
            st.markdown("---")
            st.subheader("Standard Analysis Settings")
            analysis_mode = st.radio(
                "Select Analysis Mode:",
                ("Required D0 ROAS for a Day", "Predicted Day for a Target ROAS")
            )
            
            if analysis_mode == "Required D0 ROAS for a Day":
                col1, col2 = st.columns(2)
                with col1:
                    break_even_day = st.selectbox("Select Break-Even Day:", roas_days_filtered)
                with col2:
                    margin_of_error_percent = st.slider("Margin of Error (%)", 0.0, 10.0, 5.0)

                target_roas = 1.00 # Fixed for this mode

            else:
                target_roas_percent = st.number_input(
                    "Target ROAS (%) for Prediction:", 
                    min_value=0.0, 
                    max_value=200.0, 
                    value=100.0, 
                    step=5.0
                )
                target_roas = target_roas_percent / 100
                margin_of_error_percent = 0
            
            st.markdown("---")

            if selected_campaigns:
                st.subheader("Campaign Performance Tables")

                # Table 1: Cumulative ROAS Growth from D0
                st.write("**1. Cumulative ROAS Growth from D0**")
                growth_data = []
                roas_cols_for_growth = [col for col in roas_columns_filtered if col != 'roas_d0']
                
                for campaign_name in selected_campaigns:
                    campaign_df = app_df[app_df['campaign_network'] == campaign_name]
                    campaign_roas = campaign_df[roas_columns_filtered].mean(numeric_only=True)
                    d0_roas_actual = campaign_roas['roas_d0']
                    
                    if d0_roas_actual > 0:
                        growth_row = {'Campaign': campaign_name}
                        for col in roas_cols_for_growth:
                            growth_rate = (campaign_roas[col] / d0_roas_actual) - 1
                            growth_row[f'Growth {col.replace("roas_d", "D").upper()}'] = growth_rate
                        growth_data.append(growth_row)
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data).set_index('Campaign')
                    styled_growth_df = growth_df.style.background_gradient(cmap='YlGnBu', axis=1).format('{:.2f}')
                    st.dataframe(styled_growth_df, use_container_width=True)
                else:
                    st.info("No data available to calculate growth for selected campaigns.")
                
                # Table 2: Day-over-Day ROAS Growth
                st.markdown("---")
                st.write("**2. Day-over-Day ROAS Growth**")
                
                day_over_day_growth_data = []
                intervals = roas_columns_filtered
                
                for campaign_name in selected_campaigns:
                    campaign_df = app_df[app_df['campaign_network'] == campaign_name]
                    campaign_roas = campaign_df[intervals].mean(numeric_only=True)
                    
                    d_o_d_growth_row = {'Campaign': campaign_name}
                    
                    if len(intervals) > 1:
                        for i in range(1, len(intervals)):
                            prev_day_col = intervals[i-1]
                            curr_day_col = intervals[i]
                            
                            prev_roas = campaign_roas.get(prev_day_col)
                            curr_roas = campaign_roas.get(curr_day_col)
                            
                            if prev_roas > 0:
                                d_o_d_growth = (curr_roas - prev_roas) / prev_roas
                                d_o_d_growth_row[f'Growth {prev_day_col.replace("roas_d", "D").upper()}-{curr_day_col.replace("roas_d", "D").upper()}'] = d_o_d_growth
                            else:
                                d_o_d_growth_row[f'Growth {prev_day_col.replace("roas_d", "D").upper()}-{curr_day_col.replace("roas_d", "D").upper()}'] = np.nan
                    
                    day_over_day_growth_data.append(d_o_d_growth_row)
                
                if day_over_day_growth_data:
                    d_o_d_growth_df = pd.DataFrame(day_over_day_growth_data).set_index('Campaign')
                    styled_d_o_d_growth_df = d_o_d_growth_df.style.background_gradient(cmap='YlGnBu', axis=1).format('{:.2f}')
                    st.dataframe(styled_d_o_d_growth_df, use_container_width=True)
                else:
                    st.info("No data available to calculate day-over-day growth for selected campaigns.")


                # Table 3: Actual ROAS values with heatmap
                st.markdown("---")
                st.write("**3. Actual ROAS Values**")
                
                actual_roas_data = []
                for campaign_name in selected_campaigns:
                    campaign_df = app_df[app_df['campaign_network'] == campaign_name]
                    campaign_roas = campaign_df[roas_columns_filtered].mean(numeric_only=True)
                    
                    actual_roas_row = {'Campaign': campaign_name}
                    for col in roas_columns_filtered:
                         actual_roas_row[f'ROAS {col.replace("roas_d", "D").upper()}'] = campaign_roas[col]
                    actual_roas_data.append(actual_roas_row)

                if actual_roas_data:
                    actual_roas_df = pd.DataFrame(actual_roas_data).set_index('Campaign')
                    styled_actual_roas_df = actual_roas_df.style.background_gradient(cmap='YlGnBu', axis=1).format(lambda x: f'{x*100:.2f}%' if isinstance(x, (float, np.float64)) else x)
                    st.dataframe(styled_actual_roas_df, use_container_width=True)
                else:
                    st.info("No data available to show actual ROAS for selected campaigns.")


                # Individual Campaign Break-Even Analysis section
                st.markdown("---")
                st.subheader("4. Individual Campaign Break-Even Analysis")
                
                for campaign_name in selected_campaigns:
                    st.markdown("---")
                    st.subheader(f"Campaign: {campaign_name}")
                    
                    campaign_df = app_df[app_df['campaign_network'] == campaign_name]
                    campaign_roas = campaign_df[roas_columns_filtered].mean(numeric_only=True)
                    d0_roas_actual = campaign_roas['roas_d0']
                    
                    if d0_roas_actual > 0:
                        try:
                            days = [int(col.replace('roas_d', '')) for col in roas_columns_filtered]
                            actual_roas_values = campaign_roas.values
                            
                            if analysis_mode == "Required D0 ROAS for a Day":
                                growth_multiplier = campaign_roas[f'roas_d{break_even_day}'] / d0_roas_actual
                                required_d0_roas = target_roas / growth_multiplier
                                st.write(f"**Break-Even Analysis (Target: D{break_even_day}):**")
                                st.write(f"Required D0 ROAS to break-even: `{required_d0_roas*100:.2f}%`")
                                st.write(f"Actual D0 ROAS achieved: `{d0_roas_actual*100:.2f}%`")
                                
                                difference_roas = d0_roas_actual - required_d0_roas
                                
                                if d0_roas_actual >= required_d0_roas:
                                    st.success(f"**Conclusion:** This campaign is ON TRACK! It already exceeded the required D0 ROAS by ⬆️ **{abs(difference_roas)*100:.2f}%**.")
                                elif d0_roas_actual >= required_d0_roas * (1 - margin_of_error_percent / 100):
                                    st.warning(f"**Conclusion:** This campaign is ON TRACK and within the acceptable {margin_of_error_percent}% margin. It is behind the required D0 ROAS by ➡️ **{abs(difference_roas)*100:.2f}%**.")
                                else:
                                    st.error(f"**Conclusion:** This campaign needs to improve. It is behind the required D0 ROAS by ⬇️ **{abs(difference_roas)*100:.2f}%**.")
                            else:
                                break_even_day_predicted = None
                                if any(actual_roas_values >= target_roas):
                                    x_values = np.array(days)
                                    y_values = np.array(actual_roas_values)
                                    cross_indices = np.where(y_values >= target_roas)[0]
                                    
                                    if len(cross_indices) > 0:
                                        first_cross_index = cross_indices[0]
                                        if first_cross_index == 0:
                                            break_even_day_predicted = 0.0
                                        else:
                                            x1, x2 = x_values[first_cross_index-1], x_values[first_cross_index]
                                            y1, y2 = y_values[first_cross_index-1], y_values[first_cross_index]
                                            break_even_day_predicted = x1 + (target_roas - y1) * (x2 - x1) / (y2 - y1)
                                        
                                    if break_even_day_predicted is not None and break_even_day_predicted <= days[-1]:
                                        st.success(f"**Conclusion:** This campaign is predicted to reach a **{target_roas*100:.0f}% ROAS** on or around **Day {break_even_day_predicted:.1f}**.")
                                    else:
                                        st.error(f"**Conclusion:** Based on current growth, this campaign is **not predicted** to reach a **{target_roas*100:.0f}% ROAS** within the analyzed timeframe (Day {days[-1]}).")
                            
                            st.write("\n**Daily Performance:**")
                            combined_df = pd.DataFrame({
                                'Day': [col.replace('roas_d', 'Day ') for col in roas_columns_filtered],
                                'Average ROAS': campaign_roas.values
                            })
                            
                            if analysis_mode == "Required D0 ROAS for a Day":
                                 projected_roas_values = (campaign_roas / d0_roas_actual) * required_d0_roas
                                 combined_df['Projected ROAS'] = projected_roas_values.values
                            
                            formatted_combined_df = combined_df.set_index('Day').T.style.format(lambda x: f'{x*100:.2f}%' if isinstance(x, (float, np.float64)) else x)
                            st.dataframe(formatted_combined_df, use_container_width=True)

                            # Chart is now toggleable and hidden by default
                            show_chart_for_campaign = st.checkbox(
                                f"Show Break-Even Chart for {campaign_name}",
                                value=False, # Changed to False to hide by default
                                key=f"show_chart_standard_{campaign_name}"
                            )
                            if show_chart_for_campaign:
                                fig, ax = plt.subplots(figsize=(12, 7))
                                ax.plot(days, actual_roas_values, marker='o', label='Actual Campaign ROAS')
                                ax.axhline(y=1.0, color='r', linestyle='-', label='Break-Even Point (100% ROAS)')
                                
                                if analysis_mode == "Required D0 ROAS for a Day":
                                    ax.plot(days, projected_roas_values.values, marker='x', linestyle='--', label=f'Projected ROAS for D{break_even_day} Break-Even')
                                else:
                                    ax.axhline(y=target_roas, color='b', linestyle='-', label=f'Target ROAS ({target_roas*100:.0f}%)')
                                    if break_even_day_predicted is not None and break_even_day_predicted <= days[-1]:
                                        ax.axvline(x=break_even_day_predicted, color='g', linestyle='--', label=f'Predicted Break-Even Day (~{break_even_day_predicted:.1f})')
                                
                                ax.set_title(f'ROAS Performance & Break-Even Analysis for Campaign: {campaign_name}')
                                ax.set_xlabel('Days After Installation')
                                ax.set_ylabel('Average ROAS')
                                ax.grid(True)
                                ax.legend()
                                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                                st.pyplot(fig)
                                plt.close(fig) # To prevent a warning

                        except KeyError:
                            st.error("Error: Data columns not found for analysis.")
                    else:
                        st.warning("D0 ROAS is zero, growth calculation not possible.")
                
            else:
                st.info("Please select one or more campaigns to analyze.")


        with tab2:
            st.header("Break-Even Scenario Analysis")
            st.write("Project break-even day based on optimistic and pessimistic growth scenarios.")
            
            st.markdown("---")
            st.subheader("Scenario Growth Multipliers")
            
            col1, col2 = st.columns(2)
            with col1:
                optimistic_growth_percent = st.number_input(
                    "Optimistic Growth Multiplier (%)", 
                    min_value=0, 
                    max_value=200, 
                    value=15, 
                    step=5
                )
            with col2:
                pessimistic_growth_percent = st.number_input(
                    "Pessimistic Growth Multiplier (%)", 
                    min_value=0, 
                    max_value=200, 
                    value=5, 
                    step=5
                )
            st.markdown("---")
            
            if selected_campaigns:
                for campaign_name in selected_campaigns:
                    st.markdown("---")
                    st.subheader(f"Campaign: {campaign_name}")
                    
                    campaign_df = app_df[app_df['campaign_network'] == campaign_name]
                    campaign_roas = campaign_df[roas_columns_filtered].mean(numeric_only=True)
                    d0_roas_actual = campaign_roas['roas_d0']

                    if d0_roas_actual > 0:
                        try:
                            days = [int(col.replace('roas_d', '')) for col in roas_columns_filtered]
                            actual_roas_values = campaign_roas.values
                            
                            base_roas_values = actual_roas_values.copy()
                            growth_rates = (campaign_roas / d0_roas_actual)
                            
                            optimistic_multiplier = 1 + (optimistic_growth_percent / 100)
                            optimistic_roas_values = d0_roas_actual * (1 + (growth_rates -1) * optimistic_multiplier).values
                            
                            pessimistic_multiplier = 1 - (pessimistic_growth_percent / 100)
                            pessimistic_roas_values = d0_roas_actual * (1 + (growth_rates - 1) * pessimistic_multiplier).values

                            def calculate_break_even(roas_values, days, target=1.0):
                                x_values = np.array(days)
                                y_values = np.array(roas_values)
                                if any(y_values >= target):
                                    cross_indices = np.where(y_values >= target)[0]
                                    first_cross_index = cross_indices[0]
                                    if first_cross_index == 0:
                                        return 0.0
                                    else:
                                        x1, x2 = x_values[first_cross_index-1], x_values[first_cross_index]
                                        y1, y2 = y_values[first_cross_index-1], y_values[first_cross_index]
                                        return x1 + (target - y1) * (x2 - x1) / (y2 - y1)
                                else:
                                    return None
                            
                            optimistic_breakeven = calculate_break_even(optimistic_roas_values, days)
                            base_breakeven = calculate_break_even(base_roas_values, days)
                            pessimistic_breakeven = calculate_break_even(pessimistic_roas_values, days)

                            st.write("**Predicted Break-Even Days:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if optimistic_breakeven is not None:
                                    st.success(f"**Optimistic:** Day `{optimistic_breakeven:.1f}`")
                                else:
                                    st.error(f"**Optimistic:** Not predicted within time frame")

                            with col2:
                                if base_breakeven is not None:
                                    st.info(f"**Base Case:** Day `{base_breakeven:.1f}`")
                                else:
                                    st.error(f"**Base Case:** Not predicted within time frame")

                            with col3:
                                if pessimistic_breakeven is not None:
                                    st.warning(f"**Pessimistic:** Day `{pessimistic_breakeven:.1f}`")
                                else:
                                    st.error(f"**Pessimistic:** Not predicted within time frame")
                            
                            # Chart is now toggleable and hidden by default
                            show_chart_for_campaign = st.checkbox(
                                f"Show Scenario Chart for {campaign_name}",
                                value=False, # Changed to False to hide by default
                                key=f"show_chart_scenario_{campaign_name}"
                            )

                            if show_chart_for_campaign:
                                fig, ax = plt.subplots(figsize=(12, 7))
                                ax.plot(days, optimistic_roas_values, marker='o', linestyle='--', color='green', label=f'Optimistic (+{optimistic_growth_percent}%)')
                                ax.plot(days, base_roas_values, marker='o', color='blue', label='Base Case (Historical)')
                                ax.plot(days, pessimistic_roas_values, marker='o', linestyle='--', color='red', label=f'Pessimistic (-{pessimistic_growth_percent}%)')
                                ax.axhline(y=1.0, color='black', linestyle='-', label='Break-Even Point (100% ROAS)')
                                
                                if optimistic_breakeven is not None:
                                    ax.axvline(x=optimistic_breakeven, color='green', linestyle=':', label=f'Opt. Break-Even (~{optimistic_breakeven:.1f})')
                                if base_breakeven is not None:
                                    ax.axvline(x=base_breakeven, color='blue', linestyle=':', label=f'Base Break-Even (~{base_breakeven:.1f})')
                                if pessimistic_breakeven is not None:
                                    ax.axvline(x=pessimistic_breakeven, color='red', linestyle=':', label=f'Pess. Break-Even (~{pessimistic_breakeven:.1f})')

                                ax.set_title(f'Break-Even Scenarios for Campaign: {campaign_name}')
                                ax.set_xlabel('Days After Installation')
                                ax.set_ylabel('Average ROAS')
                                ax.grid(True)
                                ax.legend()
                                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                                st.pyplot(fig)
                                plt.close(fig) # To prevent a warning

                        except KeyError:
                            st.error("Error: Data columns not found for analysis.")
                    else:
                        st.warning("D0 ROAS is zero, growth calculation not possible.")
            else:
                st.info("Please select one or more campaigns to analyze.")