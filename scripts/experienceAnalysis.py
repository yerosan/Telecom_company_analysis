import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D

class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df

    def aggregation_per_customer(self):
    
        self.df['Handset Type'] = self.df['Handset Type'].fillna(self.df['Handset Type'].mode()[0])  # Fill missing Handset Type with mode

        # Step 2: Define aggregation logic for required columns
        agg_dict = {
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Take first since handset type is likely consistent for a customer
        }

        # Step 3: Group by customer (MSISDN/Number) and aggregate the metrics
        user_experience =self.df.groupby('MSISDN/Number').agg(agg_dict).reset_index()

        # Step 4: Calculate additional metrics
        user_experience['Avg TCP Retransmission'] = (user_experience['TCP DL Retrans. Vol (Bytes)'] + user_experience['TCP UL Retrans. Vol (Bytes)']) / 2
        user_experience['Avg RTT'] = (user_experience['Avg RTT DL (ms)'] + user_experience['Avg RTT UL (ms)']) / 2
        user_experience['Avg Throughput'] = (user_experience['Avg Bearer TP DL (kbps)'] + user_experience['Avg Bearer TP UL (kbps)']) / 2

        # Final Output for Task 3.1
        print(user_experience.head())



        return user_experience
    

    def compute_top_bottom_frequent(self, customer_aggregation):

        # Step 1: Sort for Top, Bottom, and Most Frequent TCP, RTT, and Throughput values

        # Top 10 values for TCP
        top_tcp_dl = customer_aggregation['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
        bottom_tcp_dl = customer_aggregation['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
        most_frequent_tcp_dl = customer_aggregation['TCP DL Retrans. Vol (Bytes)'].value_counts().head(10)

        top_tcp_ul = customer_aggregation['TCP UL Retrans. Vol (Bytes)'].nlargest(10)
        bottom_tcp_ul = customer_aggregation['TCP UL Retrans. Vol (Bytes)'].nsmallest(10)
        most_frequent_tcp_ul = customer_aggregation['TCP UL Retrans. Vol (Bytes)'].value_counts().head(10)

        # Same for RTT
        top_rtt_dl = customer_aggregation['Avg RTT DL (ms)'].nlargest(10)
        bottom_rtt_dl = customer_aggregation['Avg RTT DL (ms)'].nsmallest(10)
        most_frequent_rtt_dl = customer_aggregation['Avg RTT DL (ms)'].value_counts().head(10)

        # Same for Throughput
        top_throughput_dl = customer_aggregation['Avg Bearer TP DL (kbps)'].nlargest(10)
        bottom_throughput_dl = customer_aggregation['Avg Bearer TP DL (kbps)'].nsmallest(10)
        most_frequent_throughput_dl = customer_aggregation['Avg Bearer TP DL (kbps)'].value_counts().head(10)

        print("Top 10 TCP DL Retransmission:", top_tcp_dl)
        print("Bottom 10 TCP DL Retransmission:", bottom_tcp_dl)
        print("Most Frequent TCP DL Retransmission:", most_frequent_tcp_dl)


    
 

    def visualize_distribution(self, customer_aggregation):
        
        handset_stats = customer_aggregation.groupby('Handset Type').agg({
            'Avg Throughput': 'mean',
            'Avg TCP Retransmission': 'mean'
        }).reset_index()

        # Step 2: Sort by Avg Throughput and select top 20 handset types
        top_20_throughput = handset_stats.sort_values(by='Avg Throughput', ascending=False).head(20)

        # Plot Distribution of Average Throughput for Top 20 Handset Types
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y='Avg Throughput', data=top_20_throughput)
        plt.title('Top 20 Handset Types by Average Throughput')
        plt.xticks(rotation=90)
        plt.show()

        # Step 3: Sort by Avg TCP Retransmission and select top 20 handset types
        top_20_tcp_retransmission = handset_stats.sort_values(by='Avg TCP Retransmission', ascending=False).head(20)

        # Plot Distribution of Average TCP Retransmission for Top 20 Handset Types
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y='Avg TCP Retransmission', data=top_20_tcp_retransmission)
        plt.title('Top 20 Handset Types by Average TCP Retransmission')
        plt.xticks(rotation=90)
        plt.show()




  

    def kmeans_clustering(self, customer_aggregation):

        # Features to be used for clustering
        features = customer_aggregation[['Avg RTT', 'Avg TCP Retransmission', 'Avg Throughput']]

        # Step 2: Standardize the features for better clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Step 3: Apply K-means clustering with k=3
        kmeans = KMeans(n_clusters=3, random_state=42)
        customer_aggregation['Cluster'] = kmeans.fit_predict(features_scaled)

        # Step 4: Analyze each cluster
        cluster_summary = customer_aggregation.groupby('Cluster').agg({
            'Avg RTT': 'mean',
            'Avg TCP Retransmission': 'mean',
            'Avg Throughput': 'mean'
        }).reset_index()

        print(cluster_summary)

        # Step 5: Plot the Clusters using a 3D Scatter Plot to visualize the clusters
       

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for each cluster, colored by cluster
        scatter = ax.scatter(
            customer_aggregation['Avg RTT'],
            customer_aggregation['Avg TCP Retransmission'],
            customer_aggregation['Avg Throughput'],
            c=customer_aggregation['Cluster'], cmap='viridis', marker='o', s=50
        )

        ax.set_xlabel('Avg RTT')
        ax.set_ylabel('Avg TCP Retransmission')
        ax.set_zlabel('Avg Throughput')
        plt.title('3D Scatter Plot of Clusters (RTT vs TCP Retransmission vs Throughput)')
        plt.colorbar(scatter)
        plt.show()

        # Step 6: 2D scatter plot for better readability (Avg RTT vs Avg Throughput colored by cluster)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Avg RTT', y='Avg Throughput', 
            hue='Cluster', data=customer_aggregation, palette='viridis', s=100
        )
        plt.title('Clusters Visualization (Avg RTT vs Avg Throughput)')
        plt.xlabel('Avg RTT')
        plt.ylabel('Avg Throughput')
        plt.show()

        # Step 7: 2D scatter plot (Avg TCP Retransmission vs Avg Throughput colored by cluster)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x='Avg TCP Retransmission', y='Avg Throughput', 
            hue='Cluster', data=customer_aggregation, palette='viridis', s=100
        )
        plt.title('Clusters Visualization (Avg TCP Retransmission vs Avg Throughput)')
        plt.xlabel('Avg TCP Retransmission')
        plt.ylabel('Avg Throughput')
        plt.show()


        return features_scaled, kmeans, 


