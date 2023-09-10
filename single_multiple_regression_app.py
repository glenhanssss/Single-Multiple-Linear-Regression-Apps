import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

# Set the page to wide mode
st.set_page_config(layout="wide")

# Title for the web app
st.title("Single/ Multiple Linear Regression Application")
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<strong>The application reads the first <em>ROW</em> in your file as the <em>Column Title (Variable Name)</em> for each <em>COLUMN</em> in the dataset. So make sure that the first <em>ROW</em> in your file is the name of each <em>Column Title (Variable Name)</em> in the dataset. Also, make sure that every cell in your file is Valid (all cells are filled and there are no duplicates) for better analysis.</strong>", unsafe_allow_html=True)
st.markdown("<strong>This application allows you to carry out linear regression analysis, both single linear regression and multiple linear regression. The analysis results obtained, namely the regression model and all the visualizations displayed, can help you predict unknown data values using other related and known data values. This really helps businesses, industry, trade, and other sectors.</strong>", unsafe_allow_html=True)

# Upload file
file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label = '')

use_defo = st.checkbox('Use example Dataset')
if use_defo:
    dataset = r'day.csv'
    st.write("[Dataset Explanation Link](https://drive.google.com/file/d/16l1-ObGv6n3j0qUwy6VyY4BOmTCzmCqO/view?usp=sharing)")

if dataset:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)

    # Display the DataFrame
    st.write("Dataset:")
    st.dataframe(df)
    
    # Select the target variable (y) and the independent variables (X)
    independent_variables = st.multiselect("Select Independent Variables (X)", df.columns.tolist())
    target_variable = st.selectbox("Select the Target Variable (y)", df.columns)
    
    if st.button("Perform Linear Regression"):
        st.markdown("<br>", unsafe_allow_html=True)
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
        st.markdown("<h3> Regression Results </h3>", unsafe_allow_html=True)
        st.write(f"Mean Squared Error = {round(mse, 4)}")
        st.write(f"R-squared (R^2) = {round(r2, 4)}")
        st.markdown('<p style="color: gray;">Notes: <br>The closer the Mean Squared Error value is to 0, the better the regression model will be. Meanwhile, the closer the R-squared value is to 1, the more accurate the regression model will be (the R-squared value is between 0 and 1).</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # count the Intercept(constant) and variable coefficients 
        st.markdown("<h3> Intercept (constant) and Variable Coefficient </h3>", unsafe_allow_html=True)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        intercept = regr.intercept_
        coefficients = regr.coef_

        # Display the Intercept(constant) and variable coefficients 
        st.write('Intercept (Constant) = \n', round(intercept, 2))        
        for feature, i in zip(independent_variables, coefficients):
            st.markdown(f'Variable Coefficient ~ <em>{(feature)}</em> = {round(i, 2)}', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Plot the predictions vs. actual values
        if len(independent_variables) == 1:

            #latex for Single Linear Regression formula
            st.markdown("<h4 style='text-align: center; color: red;'> Single Linear Regression Formula </h4>", unsafe_allow_html=True)
            st.latex(r"y = b_0 + b_1*X_1")
            st.markdown('<p style="color: gray;">y = dependent variable (target value) <br> b0 = constant <br> b1= variable coeficient <br> x1 = independent variable', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html = True)

            # Display Single Linear Regression Plot
            st.markdown("<h3> Single Linear Regression Plot </h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, color='blue', label='Training Data')
            ax.plot(X_test, y_pred, color='red', linewidth=3, label='Regression Line')
            ax.set_xlabel(f"Independent Variables(X) ~ {independent_variables}")
            ax.set_ylabel(f"Target Variable(y) ~ {target_variable}")
            ax.set_title("Training Data and Regression Line")
            ax.legend()
            st.pyplot(fig)
            # st.line_chart(
            #     pd.DataFrame({"Actual": y_test, "Predicted": y_pred}),
            #     use_container_width=True,
            # )
        else:

            #latex for formula
            st.markdown("<h4 style='text-align: center; color: red;'> Multiple Linear Regression Formula </h4>", unsafe_allow_html=True)
            st.latex(r"y = b_0 + b_1*X_1 + b_2*X_2 + ... + b_n*X_n")
            st.markdown('<p style="color: gray;">y = dependent variable (target value) <br> b0 = constant <br> b1...bn = variable coeficient <br> x1...Xn = independent variable', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html = True)
            
            #Display Multiple Linear Regression Plot Compare Actual vs Predicted
            st.markdown("<h3> Multiple Linear Regression Plot Compare Actual vs Predicted </h3>", unsafe_allow_html=True)
            st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))

            for feature in independent_variables:
                plt.figure(figsize=(8, 4))
                plt.scatter(X_test[feature], y_test, color='blue', label='Actual')
                plt.scatter(X_test[feature], y_pred, color='red', label='Predicted')
                plt.xlabel(feature)
                plt.ylabel(target_variable)
                plt.title(f"Compare Data {feature} , {target_variable}")
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

