from sklearn.metrics import pairwise_distances

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class SatisfactionAnalysis:
    def __init__(self):
        pass

    def eculidean_distance_engagement_core(self, customer_aggregation,kmeans, features_scaled):

        # Step 1: Get centroids from KMeans clustering (Task 3)
        cluster_centers = kmeans.cluster_centers_

        # Step 2: Assume the less engaged cluster is Cluster 0 and worst experience cluster is Cluster 1 (you can determine based on the analysis)
        less_engaged_cluster = 0  # This should be based on analysis, adjust as per findings
        worst_experience_cluster = 1  # This should be based on analysis

        # Step 3: Calculate the Euclidean distances for engagement score (distance from less engaged cluster)
        customer_aggregation['Engagement Score'] = pairwise_distances(
            features_scaled, [cluster_centers[less_engaged_cluster]]
        ).flatten()

        # Step 4: Calculate the Euclidean distances for experience score (distance from worst experience cluster)
        customer_aggregation['Experience Score'] = pairwise_distances(
            features_scaled, [cluster_centers[worst_experience_cluster]]
        ).flatten()

        # Step 5: Display the scores
        customer_aggregation[['Engagement Score', 'Experience Score']].head()

        return customer_aggregation
    

    def satisfaction_score(self,scoredData):
        # Calculate satisfaction score as the average of engagement and experience scores
        scoredData['Satisfaction Score'] = (scoredData['Engagement Score'] + scoredData['Experience Score']) / 2

        # Display the top 10 satisfied customers
        top_10_satisfied = scoredData.nlargest(10, 'Satisfaction Score')
        print(top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']])

    def predicting_satisfactionScore(self,scoredData):
        # Step 1: Select the features and target
        X = scoredData[['Avg RTT', 'Avg TCP Retransmission', 'Avg Throughput']]
        y = scoredData['Satisfaction Score']

        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 3: Build the regression model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)

        # Step 4: Make predictions
        y_pred = reg_model.predict(X_test)

        # Step 5: Evaluate the model using Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Step 6: Display regression coefficients
        print(f"Regression Coefficients: {reg_model.coef_}")


    def kmeans_clustring_on_Engament_and_experience_score(self, scoredData):
        # Step 1: Select Engagement and Experience Scores for clustering
        eng_exp_features = scoredData[['Engagement Score', 'Experience Score']]

        # Step 2: Apply K-means clustering with k=2
        kmeans_eng_exp = KMeans(n_clusters=2, random_state=42)
        scoredData['Satisfaction Cluster'] = kmeans_eng_exp.fit_predict(eng_exp_features)

        # Step 3: Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Engagement Score', y='Experience Score', 
            hue='Satisfaction Cluster', data=scoredData, palette='viridis', s=100
        )
        plt.title('Clustering of Users based on Engagement and Experience Scores')
        plt.show()


    def Aggregate_satisfaction_and_experiance_score_per_cluster(self, scoredData):
        # Aggregate the average satisfaction & experience score per cluster
        cluster_aggregation = scoredData.groupby('Satisfaction Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()

        return cluster_aggregation
    

    def exportingData_to_mysql(self, scoreData, db, connection):

        # Step 3: Insert data into the table
        for _, row in scoreData[['MSISDN', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].iterrows():
            
            # Ensure that numpy types are converted to native Python types
            msisdn = int(row['MSISDN'])
            engagement_score = float(row['Engagement Score'])
            experience_score = float(row['Experience Score'])
            satisfaction_score = float(row['Satisfaction Score'])
            
            query = '''
            INSERT INTO customer_scores (MSISDN, Engagement_Score, Experience_Score, Satisfaction_Score)
            VALUES (%s, %s, %s, %s)
            '''

            params = (msisdn, engagement_score, experience_score, satisfaction_score)
            
            # Execute the query using the database connection
            db.execute_query(connection, query, params)

        return  'Query executed successfully and all data added'

    


    def read_data(self, db, connection):
        try:
            # The query to fetch the data
            query = "SELECT * FROM customer_scores LIMIT 10"
            
            # Fetch data using the query
            data, columns = db.fetch_data(connection, query)
            df=pd.DataFrame(data, columns=columns)
            return df
        
        except Exception as e:
            # Rollback transaction in case of any exception
            print("Error occurred:", e)
            connection.rollback()  # Rollback the current transaction to clear the error state
        
        finally:
            db.close_connection(connection)  # Ensure the connection is closed even if there's an error








