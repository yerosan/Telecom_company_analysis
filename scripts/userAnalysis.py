from .database import DataBaseConnection
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# db=DataBaseConnection()

class UserAnalysis:
    def __init__(self):
        pass

    def creating_connection(self,db):
        connection=db.create_connection()
        return connection

    def read_data(self,db,connection):
        try:
           query="select * from xdr_data"
           data, columns= db.fetch_data(connection, query)
           df=pd.DataFrame(data, columns=columns)
           return df
        finally:
            db.close_connection(connection)

    def handleMissing(self, df):
        # Handle missing data
        # Drop columns with a large amount of missing data (> 60% missing values)
        threshold = len(df) * 0.6
        df_clean= df.dropna(thresh=threshold, axis=1)
        print("The missing____---",df_clean.shape)
        df_object=df_clean.select_dtypes(include=["object"])
        # Handling missing values in object data by replacing with a placeholder
        df_object=df_object.fillna("Unknown")

        df_non_object = df_clean.select_dtypes(exclude=['object'])

        # Impute missing values for remaining columns using median for numerical columns
        df_non_object.fillna(df_non_object.median(), inplace=True)

        df_cleaned=pd.concat([df_object, df_non_object], axis=1)

        # Drop rows with any remaining missing values in key columns
        df_cleaned.dropna(subset=['Bearer Id', 'IMSI', 'MSISDN/Number'], inplace=True)

        return df_cleaned
    

    def feature_engineering(self, df):
        # Convert 'Start' and 'End' to datetime format

        df=df[df["Start"]!="Unknown"]
        df=df[df["End"]!="Unknown"]

        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        # Drop rows where either "Start" or "End" is 'Unknown'
        # df = df[df[~((df["Start"] == 'Unknown') | (df["End"] == 'Unknown'))]]


        # Calculate session duration in seconds
        df['Session_Duration'] = (df['End'] - df['Start']).dt.total_seconds()

        # Drop unnecessary columns
        df.drop(columns=['Start', 'End', 'Start ms', 'End ms'], inplace=True)

        return df
    

    def distribution_of_sessionDuration(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Session_Duration'], bins=50, kde=True)
        plt.title('Distribution of Session Duration')
        plt.xlabel('Session Duration (seconds)')
        plt.ylabel('Frequency')
        plt.show()
    

    def correlation_matrix(self, df):
        # Plot the correlation matrix
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()
    
    def outlier_detection(self,df):
        # Visualize outliers using boxplots for key metrics
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Session_Duration'])
        plt.title('Boxplot of Session Duration')
        plt.show()

        # Check outliers for RTT (Round-Trip Time)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Avg RTT DL (ms)'])
        plt.title('Boxplot of Avg RTT DL (ms)')
        plt.show()
    

    def user_segementation(self, df):
        # Selecting key features for clustering
        X = df[['Session_Duration', 'Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']]

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df['Avg Bearer TP DL (kbps)'], y=df['Session_Duration'], hue=df['Cluster'], palette='viridis')
        plt.title('Clustering of Users Based on Engagement and Network Performance')
        plt.show()
