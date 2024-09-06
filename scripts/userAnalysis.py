from .database import DataBaseConnection
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import zscore

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


    def top_10_handset(self, df):
        top_10_handsets = df['Handset Type'].value_counts().head(10)
        # Plotting the bar chart
        plt.figure(figsize=(10, 6))
        top_10_handsets.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Handset Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print ("top_10_handsets", top_10_handsets)
        return top_10_handsets
    
    def top_3_manufacturers(self, df):
        top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
        # Plotting the bar chart
        plt.figure(figsize=(8, 6))
        top_3_manufacturers.plot(kind='bar', color='lightgreen')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Handset Manufacturer')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        print("top_3_manufacturers",top_3_manufacturers)
        
        return top_3_manufacturers
    
    def top_5HandsetPer_top3_manufacturer(self, df,  top_3_manufacturers,):
        for manufacturer in top_3_manufacturers.index:
            top_5_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
            # Plotting the bar chart for each manufacturer
            plt.figure(figsize=(8, 6))
            top_5_handsets.plot(kind='bar', color='lightcoral')
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handset Type')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            print(f"Top 5 handsets for {manufacturer}:")
            print(top_5_handsets)


    def user_behavior_analysis(self,df):
        # Aggregating per user (assuming 'MSISDN/Number' as user identifier)
        user_behavior = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',   # Number of xDR sessions
            'Dur. (ms)': 'sum',     # Total session duration
            'Total DL (Bytes)': 'sum', # Total download data
            'Total UL (Bytes)': 'sum', # Total upload data
        }).reset_index()

        # Creating total data volume per application column
        user_behavior['Total Data (Bytes)'] = user_behavior['Total DL (Bytes)'] + user_behavior['Total UL (Bytes)']
        return user_behavior




    def handleMissing(self, df):
        # Handle missing data
        # Drop columns with a large amount of missing data (> 60% missing values)
        threshold = len(df) * 0.6
        df_clean= df.dropna(thresh=threshold, axis=1)
        df_object=df_clean.select_dtypes(include=["object"])
        # Handling missing values in object data by replacing with a placeholder
        df_object=df_object.fillna("Unknown")

        df_non_object = df_clean.select_dtypes(exclude=['object'])

        # Impute missing values for remaining columns using median for numerical columns
        df_non_object.fillna(df_non_object.median(), inplace=True)

        numeric_cols=df_non_object.columns

        df_cleaned=pd.concat([df_object, df_non_object], axis=1)

        # Drop rows with any remaining missing values in key columns
        df_cleaned.dropna(subset=['Bearer Id', 'IMSI', 'MSISDN/Number'], inplace=True)

        # Handling outliers using z-score
        # numeric_cols = df.select_dtypes(include=[float, int]).columns
        z_scores = np.abs(zscore(df_cleaned[numeric_cols]))
        df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]  # Keeping rows where z-scores are less than 3


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
    


    def analyzing_basic_metrics(self, user_behavior):
        basic_stats = user_behavior.describe()
        print(basic_stats)


    def univariante_analysis(self, user_behavior):
        # Renaming specific columns
        user_behavior= user_behavior.rename(columns={'Bearer Id': 'Session Frequency', 'Total Data (Bytes)': 'Total Data Volume (Bytes)', 'Dur. (ms)':'Total Duration (ms)'})

        dispersion_params = user_behavior[['Session Frequency', 'Total Data Volume (Bytes)', 'Total Duration (ms)']].std()
        print("Dispersion Parameters:\n", dispersion_params)


    def graphic_univariant_analysis(self, user_behavior):

        # Histogram of total data volume

        user_behavior= user_behavior.rename(columns={'Bearer Id': 'Session Frequency', 'Total Data (Bytes)': 'Total Data Volume (Bytes)', 'Dur. (ms)':'Total Duration (ms)'})

        sns.histplot(user_behavior['Total Data Volume (Bytes)'], kde=True)
        plt.title('Total Data Volume Distribution')
        plt.show()

        # Boxplot for session frequency
        sns.boxplot(user_behavior['Session Frequency'])
        plt.title('Session Frequency Distribution')
        plt.show()

    def bivariante_analysis(self, user_behavior):
        # Scatter plot between total data volume and session duration

        user_behavior= user_behavior.rename(columns={'Bearer Id': 'Session Frequency', 'Total Data (Bytes)': 'Total Data Volume (Bytes)', 'Dur. (ms)':'Total Duration (ms)'})

        sns.scatterplot(x='Total Duration (ms)', y='Total Data Volume (Bytes)', data=user_behavior)
        plt.title('Total Duration vs Total Data Volume')
        plt.show()

    def correlation_analysis(self, user_behavior):
        # Correlation matrix for specific apps data
        correlation_matrix = user_behavior[['Social Media DL (Bytes)', 'Google DL (Bytes)', 
                                            'Email DL (Bytes)', 'Youtube DL (Bytes)', 
                                            'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
                                            'Other DL (Bytes)']].corr()

        print(correlation_matrix)





    

    

    def distribution_of_sessionDuration(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Session_Duration'], bins=50, kde=True)
        plt.title('Distribution of Session Duration')
        plt.xlabel('Session Duration (seconds)')
        plt.ylabel('Frequency')
        plt.show()
    
    def dimensionalReduction_using_pca(self, user_behavior):

        # Standardize the data before PCA
        X = user_behavior[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 
                        'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]
        X_scaled = StandardScaler().fit_transform(X)

        # Applying PCA
        pca = PCA(n_components=2)  # Reduce to 2 dimensions
        pca_result = pca.fit_transform(X_scaled)

        # Plot PCA result
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title('PCA Dimensionality Reduction')
        plt.show()

        # Explained variance ratio (how much info we retained)
        print(pca.explained_variance_ratio_)
        return pca_result

    
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
    
    def user_segementation_basedOn_handset(self, df, top_10_handsets):
        # Segment users based on handset type
        user_segments =df.groupby(['Handset Type', 'Handset Manufacturer']).agg({
            # 'xDR_sessions': 'sum',
            'Session_Duration': 'mean',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()

        print("User Segments:\n", user_segments)

        # Profile the top segments
        top_segments = user_segments[user_segments['Handset Type'].isin(top_10_handsets.index)]
        print("Top User Segments Profile:\n", top_segments)


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
