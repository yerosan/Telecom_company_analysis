import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from scripts.database import DataBaseConnection
from scripts.userAnalysis import UserAnalysis

# Load the dataset
@st.cache_resource
def load_data():
    db=DataBaseConnection()
    user_analysis=UserAnalysis()
    connection=user_analysis.creating_connection(db)
    data=db.read_data(db, connection)
    # Example data - replace with your actual dataset
    return data

# Main Dashboard Layout
def main():
    st.title("User Engagement and Experience Dashboard")
    
    # Load the data
    data = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose a view", ['Overview', 'Clustering', 'Regression', 'Conclusion'])
    
    if options == 'Overview':
        st.header("Project Overview")
        st.write("""
            - Task 1: Initial data exploration and analysis of user behavior.
            - Task 2: Segmentation based on user experience metrics.
            - Task 3: Satisfaction prediction using regression.
            - Task 4: Model deployment and tracking.
        """)
        st.write("Below is a sample of the data used:")
        st.dataframe(data.head())

    elif options == 'Clustering':
        st.header("Clustering: Experience and Engagement Analysis")
        st.write("This scatter plot shows how users are clustered based on their engagement and experience scores.")
        
        # Scatter plot for Clustering (Task 3.4)
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Engagement Score', y='Experience Score', hue='Cluster', palette='viridis')
        plt.title("User Clusters Based on Engagement and Experience")
        plt.xlabel("Engagement Score")
        plt.ylabel("Experience Score")
        st.pyplot(fig)
        
    elif options == 'Regression':
        st.header("Satisfaction Prediction Model")
        
        # Input features for the model
        X = data[['engagement_score', 'experience_score']]
        y = data['satisfaction_score']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the Random Forest Model
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Display the metrics
        st.write(f"Mean Squared Error of the model: {mse}")
        
        # Plot the actual vs predicted values
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions)
        plt.plot([0, max(y_test)], [0, max(y_test)], '--', color='red', label='Ideal')
        plt.title("Actual vs Predicted Satisfaction Scores")
        plt.xlabel("Actual Satisfaction Score")
        plt.ylabel("Predicted Satisfaction Score")
        plt.legend()
        st.pyplot(fig)
        
    elif options == 'Conclusion':
        st.header("Conclusion & Recommendations")
        st.write("""
            **Overall Project Conclusion**:
            - Task 1: Identified key user experience factors.
            - Task 2: Clustered users based on engagement and experience metrics.
            - Task 3: Satisfaction prediction was accurate, offering valuable insights for improving customer experience.
            - Task 4: Model deployment ensures continuous monitoring and improvement.

            **Key Recommendations**:
            1. Focus on users in low-experience clusters.
            2. Improve engagement strategies for low-scoring users.
            3. Use the deployed model to track performance in real-time.
        """)
        
if __name__ == '__main__':
    main()
