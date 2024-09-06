from .database import DataBaseConnection
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from scipy.stats import zscore


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



class UserEngagement:
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
        
    def feature_engineering(self, df):
        # Convert 'Start' and 'End' to datetime format

        df=df[df["Start"]!="Unknown"]
        df=df[df["End"]!="Unknown"]

        df['Start'] = pd.to_datetime(df['Start'])
        df['End'] = pd.to_datetime(df['End'])
        # Drop rows where either "Start" or "End" is 'Unknown'
        # df = df[df[~((df["Start"] == 'Unknown') | (df["End"] == 'Unknown'))]]


    

    def engagement_metrics(self, df):

        # Aggregate user engagement metrics
        user_engagement = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',          # Session frequency
            'Dur. (ms)': 'sum',            # Total session duration
            'Total DL (Bytes)': 'sum',     # Total download data
            'Total UL (Bytes)': 'sum'      # Total upload data
        }).reset_index()

        # Calculate total traffic (download + upload)
        user_engagement['Total Traffic (Bytes)'] = user_engagement['Total DL (Bytes)'] + user_engagement['Total UL (Bytes)']

        # Rename columns for clarity
        user_engagement.columns = ['MSISDN', 'Session Frequency', 'Total Duration (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Total Traffic (Bytes)']

        return user_engagement
    

    def top_10_userByEngagement(self,user_engagement):

        # Top 10 customers by session frequency
        top_10_frequency = user_engagement[['MSISDN', 'Session Frequency']].sort_values(by='Session Frequency', ascending=False).head(10)

        # Top 10 customers by session duration
        top_10_duration = user_engagement[['MSISDN', 'Total Duration (ms)']].sort_values(by='Total Duration (ms)', ascending=False).head(10)

        # Top 10 customers by total traffic
        top_10_traffic = user_engagement[['MSISDN', 'Total Traffic (Bytes)']].sort_values(by='Total Traffic (Bytes)', ascending=False).head(10)

        print("Top 10 Customers by Session Frequency:\n", top_10_frequency)
        print("Top 10 Customers by Session Duration:\n", top_10_duration)
        print("Top 10 Customers by Total Traffic:\n", top_10_traffic)






    def normalizatoin_and_clusteringEngagement(self, user_engagement):

        # Select the metrics to normalize
        metrics = user_engagement[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']]

        # Normalize using Min-Max scaling
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(metrics)

        # Convert back to a DataFrame for clarity
        normalized_df = pd.DataFrame(normalized_metrics, columns=['Normalized Frequency', 'Normalized Duration', 'Normalized Traffic'])

        # Applying K-means clustering (k=3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        user_engagement['Cluster'] = kmeans.fit_predict(normalized_metrics)

        # Add cluster labels to the normalized metrics
        user_engagement['Cluster'] = kmeans.labels_

        # Aggregating metrics per cluster
        cluster_metrics = user_engagement.groupby('Cluster').agg({
            'Session Frequency': ['min', 'max', 'mean', 'sum'],
            'Total Duration (ms)': ['min', 'max', 'mean', 'sum'],
            'Total Traffic (Bytes)': ['min', 'max', 'mean', 'sum']
        })

        # print(cluster_metrics)

        


        # Plot distribution of metrics by cluster
        sns.boxplot(x='Cluster', y='Session Frequency', data=user_engagement)
        plt.title('Session Frequency per Cluster')
        plt.show()

        sns.boxplot(x='Cluster', y='Total Duration (ms)', data=user_engagement)
        plt.title('Session Duration per Cluster')
        plt.show()

        sns.boxplot(x='Cluster', y='Total Traffic (Bytes)', data=user_engagement)
        plt.title('Total Traffic per Cluster')
        plt.show()
        
        

        inertia = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_metrics)
            inertia.append(kmeans.inertia_)

        # Plot the Elbow graph
        plt.plot(range(1, 11), inertia, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()

        return cluster_metrics, normalized_metrics
    



    def app_traffic(self,df):
        app_traffic = df.groupby('MSISDN/Number').agg({
            'Social Media DL (Bytes)': 'sum', 'Social Media UL (Bytes)': 'sum',
            'Google DL (Bytes)': 'sum', 'Google UL (Bytes)': 'sum',
            'Email DL (Bytes)': 'sum', 'Email UL (Bytes)': 'sum',
            'Youtube DL (Bytes)': 'sum', 'Youtube UL (Bytes)': 'sum',
            'Netflix DL (Bytes)': 'sum', 'Netflix UL (Bytes)': 'sum',
            'Gaming DL (Bytes)': 'sum', 'Gaming UL (Bytes)': 'sum',
            'Other DL (Bytes)': 'sum', 'Other UL (Bytes)': 'sum'
        }).reset_index()

        # Calculate total traffic per application
        app_traffic['Total Social Media Traffic'] = app_traffic['Social Media DL (Bytes)'] + app_traffic['Social Media UL (Bytes)']
        app_traffic['Total Google Traffic'] = app_traffic['Google DL (Bytes)'] + app_traffic['Google UL (Bytes)']
        app_traffic['Total Youtube Traffic'] = app_traffic['Youtube DL (Bytes)'] + app_traffic['Youtube UL (Bytes)']

        return app_traffic
    
    def top_10_mostEngage_User_PerApplication(self, app_traffic):
        top_10_social_media = app_traffic[['MSISDN/Number', 'Total Social Media Traffic']].sort_values(by='Total Social Media Traffic', ascending=False).head(10)
        top_10_google = app_traffic[['MSISDN/Number', 'Total Google Traffic']].sort_values(by='Total Google Traffic', ascending=False).head(10)
        top_10_youtube = app_traffic[['MSISDN/Number', 'Total Youtube Traffic']].sort_values(by='Total Youtube Traffic', ascending=False).head(10)

        print("Top 10 Most Engaged Users in Social Media:\n", top_10_social_media)
        print("Top 10 Most Engaged Users in Google:\n", top_10_google)
        print("Top 10 Most Engaged Users in YouTube:\n", top_10_youtube)


    def top_3Most_userApplication(sef, app_traffic):
        # Plot top 3 most used applications
        top_apps = app_traffic[['Total Social Media Traffic', 'Total Google Traffic', 'Total Youtube Traffic']].sum()

        top_apps.plot(kind='bar', color=['blue', 'green', 'red'])
        plt.title('Top 3 Most Used Applications')
        plt.ylabel('Total Traffic (Bytes)')
        plt.xticks(rotation=45)
        plt.show()







    def application_spacificEngagement(self, df):
        # Aggregate user traffic per application
        app_engagement = df.groupby(['Bearer Id', 'Social Media DL (Bytes)']).agg({
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()

        # Calculate total traffic per application
        app_engagement['total_traffic'] = app_engagement['Total DL (Bytes)'] + app_engagement['Total UL (Bytes)']

        # Identify top 3 most used applications
        top_apps = app_engagement.groupby('Social Media DL (Bytes)')['total_traffic'].sum().nlargest(3).index

        # Visualize the top 3 applications
        for app in top_apps:
            app_data = app_engagement[app_engagement['Social Media DL (Bytes)'] == app]
            sns.barplot(x='Bearer Id', y='total_traffic', data=app_data.sort_values(by='total_traffic', ascending=False).head(10))
            plt.title(f'Top 10 Users for {app}')
            plt.xticks(rotation=45)
            plt.show()

            # app_data = app_engagement[app_engagement['Youtube DL (Bytes)'] == app]
            # sns.barplot(x='Bearer Id', y='total_traffic', data=app_data.sort_values(by='total_traffic', ascending=False).head(10))
            # plt.title(f'Top 10 Users for {app}')
            # plt.xticks(rotation=45)
            # plt.show()

