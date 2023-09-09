import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set the page to wide mode
st.set_page_config(layout="wide")

# Title for the web app
st.title("Single/ Multiple Linear Regression Application")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("Uploaded Data:")
    st.write(df, width=9999)

    # Select the target variable (y) and the independent variables (X)
    independent_variables = st.multiselect("Select Independent Variables (X)", df.columns.tolist())
    target_variable = st.selectbox("Select the Target Variable (y)", df.columns)
    
    if st.button("Perform Linear Regression"):
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Split the data into training and testing sets
        X = df[independent_variables]
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display regression results
        st.write("Regression Results:")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared (R2): {r2}")
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Plot the predictions vs. actual values
        if len(independent_variables) == 1:
            st.write("Single Linear Regression Plot:")
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, color='blue', label='Training Data')
            ax.plot(X_test, y_pred, color='red', linewidth=3, label='Linear Regression Line')
            ax.set_xlabel("Independent Variables (X)")
            ax.set_ylabel("Target Variable (y)")
            ax.set_title("Regresi Linear - Data Latih dan Garis Regresi")
            st.pyplot(fig)
            # st.line_chart(
            #     pd.DataFrame({"Actual": y_test, "Predicted": y_pred}),
            #     use_container_width=True,
            # )
        else: 
            st.write("Multiple Linear Regression Plot Compare Actual vs Predicted:")
            st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))

            for feature in independent_variables:
                plt.figure(figsize=(8, 4))
                plt.scatter(X_test[feature], y_test, color='blue', label='Actual')
                plt.scatter(X_test[feature], y_pred, color='red', label='Predicted')
                plt.xlabel(feature)
                plt.ylabel(target_variable)
                plt.title(f"Compare Data for {feature} vs {target_variable}")
                plt.legend()
                st.pyplot(plt)

            # st.write("Multiple Linear Regression Plot:")
            # for feature in independent_variables:
            #     plt.scatter(X_test[feature], y_test, label=f"Actual vs. {feature}", alpha=0.6)
            #     plt.plot(X_test[feature], y_pred, label=f"Regression Line ({feature})")
            # plt.xlabel("Feature Values")
            # plt.ylabel("Target Variable")
            # plt.title("Regression Lines for Selected Features")
            # plt.legend()
            # plt.tight_layout()
            # st.pyplot(plt)
