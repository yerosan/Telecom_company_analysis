# Telecom Company Analysis

This project focuses on analyzing a telecom company's customer data to gain insights into user behavior, engagement, and experience. It includes tasks such as user overview analysis, engagement clustering, satisfaction score calculations, and more to support better decision-making for network resource allocation and customer experience improvements.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Analysis Tasks](#data-analysis-tasks)
- [Project Structure](#project-structure)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The main objective of this project is to help a telecom company improve its network performance, customer engagement, and satisfaction through detailed data analysis. By analyzing telecom datasets, key metrics such as session frequency, data usage, TCP retransmissions, and throughput are analyzed to classify users, predict satisfaction scores, and generate actionable insights.

### Key Objectives:
- **User Overview Analysis:** Analyze user behaviors by examining handset types, application usage, and total data consumption.
- **User Engagement Analysis:** Use clustering techniques to segment users based on session frequency, duration, and traffic.
- **Experience & Satisfaction Analysis:** Analyze network experience metrics like TCP retransmissions and throughput to compute satisfaction scores.

## Data Analysis Tasks
The project is structured into the following tasks:

- **Task 1: User Overview Analysis**
  - Identify the top 10 handsets and top 3 handset manufacturers.
  - Aggregate per-user statistics on application usage and data consumption.
  - Visualize the data to reveal patterns.

- **Task 2: User Engagement Analysis**
  - Analyze session frequency, session duration, and total traffic per user.
  - Apply k-means clustering to classify customers into different engagement levels.
  - Visualize application usage trends.

- **Task 3: Experience Analysis**
  - Analyze key experience metrics such as TCP retransmission, RTT, and throughput.
  - Segment users into experience-based clusters.

- **Task 4: Satisfaction Analysis**
  - Calculate engagement and experience scores for each user.
  - Predict customer satisfaction using regression modeling.
  - Deploy the satisfaction prediction model using Docker or other ML Ops tools.

## Project Structure
```bash
Telecom_company_analysis/
├── data/                     # Contains datasets used for analysis
├── notebooks/                # Jupyter notebooks with detailed analysis
├── src/                      # Python scripts for data processing and modeling
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```
Results
Some key insights derived from the analysis include:

Top Handsets: Identifying the most popular handset types to guide marketing strategies.
Application Usage: Analyzing data consumption patterns and session duration to optimize network resource allocation.
User Clustering: Grouping users by engagement and experience for personalized service.
Satisfaction Prediction: Predicting satisfaction scores to enhance customer retention.
Requirements
The following dependencies are required for this project:

Python 3.x
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
SQLAlchemy (for MySQL integration)
Docker (for model deployment)

Install the required libraries using:
```bash
pip install -r requirements.txt
```

Installation
To install and run the project locally, follow these steps:

Clone the repository:

```bash
git clone https://github.com/yerosan/Telecom_company_analysis.git
```
Navigate to the project directory:

```bash
cd Telecom_company_analysis
Install the required dependencies:
```
```bash
pip install -r requirements.txt
```
To run the project:

- Data Preparation: Place the telecom dataset in the data/ folder.
- Run Analysis: Use the Jupyter notebooks in the notebooks/ folder or execute the Python scripts in the src/ folder.
- Visualization: Visualizations and plots will be generated for insights into user engagement and experience.
- Model Deployment: Follow Task 4 for deploying the satisfaction prediction model using Docker.

This project is licensed under the MIT License. See the LICENSE file for more details.
