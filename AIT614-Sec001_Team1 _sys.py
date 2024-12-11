# Databricks notebook source
# -----------------------------------------------------------------------------
# Final Project
# Team 1
# Course: AIT614_001
# -----------------------------------------------------------------------------
# Team Members:
# 1. Suraj Poldas
# 2. Mano Sappa Harsha
# 3. Vasishta Chandala
# 4. Mohammed Tareq Sajjad Ali
# 5. Saivarun Tanjore Raghavendra
# -----------------------------------------------------------------------------


# COMMAND ----------

# MAGIC %pip install imbalanced-learn

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/FileStore/tables/'))

# COMMAND ----------

df = spark.read.format("csv").option("header","true").load("dbfs:/FileStore/tables/dataset_cos_in.csv")

# COMMAND ----------

from pyspark.sql.functions import col, count, isnan, when, mean, stddev, min, max

row_count = df.count()
column_count = len(df.columns)

print(f"Number of Rows: {row_count}")
print(f"Number of Columns: {column_count}")

# COMMAND ----------


# Retrieving and display column names
column_names = df.columns
print("Column Names:")
for name in column_names:
    print(name)

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count

# Counting null values in each column
null_counts = df.select([
    count(when(col(f"`{c}`").isNull() | isnan(f"`{c}`"), c)).alias(c) for c in df.columns
])

# Displaying the result
display(null_counts)



# COMMAND ----------

# MAGIC %md
# MAGIC Checking columns and their datatypes

# COMMAND ----------

# Creating mapping of original column names to simplified names
renamed_columns = {
    '1 - State Code': 'State_Code',
    '1 - State Name': 'State_Name',
    '8 - Structure Number': 'Structure_Number',
    '22 - Owner Agency': 'Owner_Agency',
    '3 - County Code': 'County_Code',
    '3 - County Name': 'County_Name',
    '27 - Year Built': 'Year_Built',
    '29 - Average Daily Traffic': 'Average_Daily_Traffic',
    '43A - Main Span Material': 'Main_Span_Material',
    '43B - Main Span Design': 'Main_Span_Design',
    '49 - Structure Length (ft.)': 'Structure_Length_ft',
    '6A - Features Intersected': 'Features_Intersected',
    '7 - Facility Carried By Structure': 'Facility_Carried_By_Structure',
    'CAT10 - Bridge Condition': 'Bridge_Condition',
    'Bridge Age (yr)': 'Bridge_Age_yr',
    'CAT29 - Deck Area (sq. ft.)': 'Deck_Area_sq_ft',
    '9 - Location': 'Location',
    '16 - Latitude (decimal)': 'Latitude',
    '17 - Longitude (decimal)': 'Longitude',
    '60 - Substructure Condition Rating': 'Substructure_Condition_Rating',
    '59 - Superstructure Condition Rating': 'Superstructure_Condition_Rating',
    '109 - Average Daily Truck Traffic (Percent ADT)': 'Average_Daily_Truck_Traffic_Percent',
    '114 - Future Average Daily Traffic': 'Future_Average_Daily_Traffic',
    '115 - Year of Future Average Daily Traffic': 'Year_of_Future_Average_Daily_Traffic',
    '96 - Total Project Cost': 'Total_Project_Cost',
    '94 - Bridge Improvement Cost': 'Bridge_Improvement_Cost',
    '95 - Roadway Improvement Cost': 'Roadway_Improvement_Cost',
    '97 - Year of Improvement Cost Estimate': 'Year_of_Improvement_Cost_Estimate',
    'Computed - Average Daily Truck Traffic (Volume)': 'Computed_Average_Daily_Truck_Traffic_Volume',
    'Average Relative Humidity': 'Average_Relative_Humidity',
    'Average Temperature': 'Average_Temperature',
    'Maximum Temperature': 'Maximum_Temperature',
    'Mean Wind Speed': 'Mean_Wind_Speed',
    'Minimum Temperature': 'Minimum_Temperature',
    'Number of Freeze-Thaw Cycles': 'Number_of_Freeze_Thaw_Cycles',
    'Number of Snowfall Days': 'Number_of_Snowfall_Days',
    'Number of Days with Measurable Precipitation': 'Number_of_Days_with_Measurable_Precipitation',
    'Number of Days with Temperature Below 0?C': 'Number_of_Days_with_Temperature_Below_0C',
    'Prevailing Wind Direction': 'Prevailing_Wind_Direction',
    'Time of Wetness': 'Time_of_Wetness',
    'Total Precipitation': 'Total_Precipitation'
}

# Renaming columns
for old_name, new_name in renamed_columns.items():
    df = df.withColumnRenamed(old_name, new_name)

display(df)

# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType, DoubleType
from pyspark.sql.functions import col

# Defining the mapping of simplified column names to target datatypes
column_type_mapping = {
    'State_Code': IntegerType(),
    'Year_Built': IntegerType(),
    'Average_Daily_Traffic': IntegerType(),
    'Structure_Length_ft': FloatType(),
    'Deck_Area_sq_ft': FloatType(),
    'Bridge_Age_yr': IntegerType(),
    'Latitude': FloatType(),
    'Longitude': FloatType(),
    'Average_Daily_Truck_Traffic_Percent': FloatType(),
    'Future_Average_Daily_Traffic': IntegerType(),
    'Year_of_Future_Average_Daily_Traffic': IntegerType(),
    'Total_Project_Cost': FloatType(),
    'Bridge_Improvement_Cost': FloatType(),
    'Roadway_Improvement_Cost': FloatType(),
    'Average_Relative_Humidity': FloatType(),
    'Average_Temperature': FloatType(),
    'Maximum_Temperature': FloatType(),
    'Mean_Wind_Speed': FloatType(),
    'Minimum_Temperature': FloatType(),
    'Number_of_Freeze_Thaw_Cycles': IntegerType(),
    'Number_of_Snowfall_Days': IntegerType(),
    'Number_of_Days_with_Measurable_Precipitation': IntegerType(),
    'Number_of_Days_with_Temperature_Below_0C': IntegerType(),
    'Total_Precipitation': FloatType()
}

# Converting each column to the appropriate type
for column, dtype in column_type_mapping.items():
    df = df.withColumn(column, col(column).cast(dtype))

df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC **Checking percentage of missing values**

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count

# Calculate the total number of rows
total_rows = df.count()

# Calculate missing values and their percentage for each column
missing_percentage = df.select([
    ((count(when(col(f"`{c}`").isNull() | isnan(f"`{c}`"), c)) / total_rows) * 100).alias(c) for c in df.columns
])

# Display the result
display(missing_percentage)


# COMMAND ----------

# MAGIC %md
# MAGIC **Removing missing values which are greater than 30%**

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count

total_rows = df.count()

# Calculating the percentage of missing values for each column
missing_percentage = df.select([
    (count(when(col(f"`{c}`").isNull() | isnan(f"`{c}`"), c)) / total_rows * 100).alias(c) for c in df.columns
])

# Collecting the missing percentage as a dictionary
missing_percentage_dict = missing_percentage.collect()[0].asDict()

# Identifying columns with more than 30% missing values
columns_to_drop = [col_name for col_name, perc in missing_percentage_dict.items() if perc > 30]

df_cleaned = df.drop(*columns_to_drop)

df_cleaned.printSchema()


# COMMAND ----------

#finding which columns have missing values for remaining imputation
from pyspark.sql.functions import col, isnan, when, count

# Checking for missing values in each column
remaining_missing_values = df_cleaned.select([
    count(when(col(f"`{c}`").isNull() | isnan(f"`{c}`"), c)).alias(c) for c in df_cleaned.columns
])

# Displaying columns with non-zero missing values
remaining_missing_values_df = remaining_missing_values.toPandas()
columns_with_missing_values = remaining_missing_values_df.loc[:, (remaining_missing_values_df > 0).any()].T
columns_with_missing_values.columns = ['Missing Count']
print(columns_with_missing_values)


# COMMAND ----------

from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import col, when, lit, mean

numeric_conversion_columns = [
    'Total_Project_Cost', 'Bridge_Improvement_Cost', 'Year_of_Improvement_Cost_Estimate'
]
for col_name in numeric_conversion_columns:
    df_cleaned = df_cleaned.withColumn(col_name, col(col_name).cast(FloatType()))

# Imputing numeric columns
for col_name in numeric_conversion_columns:
    # Calculating the median (approxQuantile for performance)
    median_value = (
        df_cleaned.stat.approxQuantile(col_name, [0.5], 0.1)[0] 
    )
    df_cleaned = df_cleaned.withColumn(
        col_name,
        when(col(col_name).isNull(), lit(median_value)).otherwise(col(col_name))
    )

# Verifying remaining missing values
remaining_missing_values = df_cleaned.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df_cleaned.columns
])



# COMMAND ----------

from pyspark.sql.functions import col, when, lit, mean, expr

numeric_columns = [
    'Average_Daily_Truck_Traffic_Percent', 'Future_Average_Daily_Traffic',
    'Year_of_Future_Average_Daily_Traffic', 'Total_Project_Cost',
    'Bridge_Improvement_Cost', 'Year_of_Improvement_Cost_Estimate',
    'Computed_Average_Daily_Truck_Traffic_Volume', 'Average_Relative_Humidity',
    'Average_Temperature', 'Maximum_Temperature', 'Mean_Wind_Speed',
    'Minimum_Temperature', 'Number_of_Days_with_Measurable_Precipitation',
    'Number_of_Days_with_Temperature_Below_0C', 'Total_Precipitation'
]
for col_name in numeric_columns:
    df_cleaned = df_cleaned.withColumn(col_name, col(col_name).cast(FloatType()))

for col_name in numeric_columns:
    mean_value = df_cleaned.select(mean(col(col_name)).alias("mean")).first()[0]
    df_cleaned = df_cleaned.withColumn(
        col_name,
        when(col(col_name).isNull(), lit(mean_value)).otherwise(col(col_name))
    )

# Handling nominal columns (if any exist) 
nominal_columns = ['Prevailing_Wind_Direction', 'Time_of_Wetness'] 
for col_name in nominal_columns:
    mode_value = (
        df_cleaned.groupBy(col_name)
        .count()
        .orderBy(expr("count").desc())
        .first()[0]
    )
    df_cleaned = df_cleaned.withColumn(
        col_name,
        when(col(col_name).isNull(), lit(mode_value)).otherwise(col(col_name))
    )

remaining_missing_values = df_cleaned.select([
    count(when(col(c).isNull(), c)).alias(c) for c in df_cleaned.columns
])


# COMMAND ----------

# Number of rows
num_rows = df_cleaned.count()

# Number of columns
num_columns = len(df_cleaned.columns)

print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")


# COMMAND ----------

from pyspark.sql.functions import col, trim, regexp_replace

# Cleaning columns: Removing single quotes and trim spaces
for column in df_cleaned.columns:
    df_cleaned = df_cleaned.withColumn(column, regexp_replace(trim(col(column)), "'", ""))


# COMMAND ----------

# Identifying columns with floating-point values and rounding them to 2 decimal places
float_columns = [column for column, dtype in df_cleaned.dtypes if dtype in ('double', 'float')]

for column in float_columns:
    df_cleaned = df_cleaned.withColumn(column, round(col(column), 2))

# COMMAND ----------

#------##Initial preprocessing is done##--------#

# COMMAND ----------

## Preprocessing for Irrelevant values
## EDA - Begins

# COMMAND ----------

from pyspark.sql.functions import col, when, lit

valid_conditions = ["Good", "Fair", "Poor"]

df_cleaned = df_cleaned.withColumn(
    "Bridge_Condition",
    when(col("Bridge_Condition").isin(valid_conditions), col("Bridge_Condition")).otherwise(None)
)

# Calculating the mode for valid bridge conditions
mode_value = (
    df_cleaned.groupBy("Bridge_Condition")
    .count()
    .orderBy(col("count").desc())
    .first()[0]
)

# Imputing missing values with the mode
df_cleaned = df_cleaned.withColumn(
    "Bridge_Condition",
    when(col("Bridge_Condition").isNull(), lit(mode_value)).otherwise(col("Bridge_Condition"))
)





# COMMAND ----------

df_cleaned.groupBy("Bridge_Condition").count().display()

# COMMAND ----------

from pyspark.sql.functions import col, when

# Creating a new column 'Location_Type' to classify as 'Coastal' or 'Inland'
df_cleaned = df_cleaned.withColumn(
    "Location_Type",
    when(col("Longitude") > -78.0, "Coastal").otherwise("Inland")
)

# Verify the results
df_cleaned.select("Latitude", "Longitude", "Location_Type").show(truncate=False)


# COMMAND ----------

display(df_cleaned)

# COMMAND ----------

## How Does bridge age coorelated with bridge condition
df_cleaned.groupBy("Bridge_Condition").agg({"Bridge_Age_yr": "avg"}).display()


# COMMAND ----------

## % of coastal vs inland
df_cleaned.groupBy("Location_Type").count().display()


# COMMAND ----------

## Which material is most commonly used in main span?
df_cleaned.groupBy("Main_Span_Material").count().orderBy("count", ascending=False).display()


# COMMAND ----------

## Are coastal bridge older on average compared to inland bridges?
df_cleaned.groupBy("Location_Type").agg({"Bridge_Age_yr": "avg"}).display()


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import FloatType

# Convert necessary columns to FloatType
df_cleaned = df_cleaned.withColumn("Average_Temperature", col("Average_Temperature").cast(FloatType()))
df_cleaned = df_cleaned.withColumn("Average_Relative_Humidity", col("Average_Relative_Humidity").cast(FloatType()))
df_cleaned = df_cleaned.withColumn("Total_Precipitation", col("Total_Precipitation").cast(FloatType()))

# Verify schema after conversion
df_cleaned.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC **Determining K-Clusters using Elbow Method**

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

df_cleaned = df_cleaned.dropna(subset=["Average_Daily_Traffic", "Bridge_Age_yr", "Average_Temperature", "Total_Precipitation"])

df_cleaned = df_cleaned.withColumn("Average_Daily_Traffic", col("Average_Daily_Traffic").cast("float")) \
                       .withColumn("Bridge_Age_yr", col("Bridge_Age_yr").cast("float")) \
                       .withColumn("Average_Temperature", col("Average_Temperature").cast("float")) \
                       .withColumn("Total_Precipitation", col("Total_Precipitation").cast("float"))

vector_assembler = VectorAssembler(
    inputCols=["Average_Daily_Traffic", "Bridge_Age_yr", "Average_Temperature", "Total_Precipitation"],
    outputCol="features"
)
df_vectorized = vector_assembler.transform(df_cleaned)

# Step 4: Elbow Method
cost = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features")
    model = kmeans.fit(df_vectorized)
    cost.append(model.summary.trainingCost)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, cost, marker='o', linestyle='-')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(k_values)
plt.grid()
plt.show()

# K-Means with k=4
kmeans = KMeans().setK(4).setSeed(42).setFeaturesCol("features")
kmeans_model = kmeans.fit(df_vectorized)

df_clustered = kmeans_model.transform(df_vectorized)

# Evaluating Clustering with Silhouette Score
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette_score = evaluator.evaluate(df_clustered)
print(f"Silhouette Score: {silhouette_score:.2f}")

# Displaying Cluster Centers
cluster_centers = kmeans_model.clusterCenters()
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i} Center: {center}")


# COMMAND ----------

feature_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']

# Converting PySpark DataFrame to Pandas DataFrame for scikit-learn
X = df_cleaned.select(feature_columns)
X_pandas = X.toPandas()  


# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pandas)  # Scaling the features


# COMMAND ----------

from sklearn.cluster import KMeans

# Defining number of clusters
n_clusters = 4

# Applying KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fitting the model and predicting the cluster labels
y_kmeans = kmeans.fit_predict(X_scaled)  # y_kmeans contains the cluster labels


# COMMAND ----------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Applying PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the clusters in 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.title("2D PCA of Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()


# COMMAND ----------

# Checking explained variance ratio for each principal component
print(f"Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")


# COMMAND ----------

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Applying PCA for dimensionality reduction to 3D
pca_3d = PCA(n_components=4)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Plotting the clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y_kmeans, cmap='viridis')
ax.set_title("3D PCA of Clusters")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.colorbar(scatter)
plt.show()


# COMMAND ----------

from sklearn.decomposition import PCA

# Applying PCA for dimensionality reduction to 3 components
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Explained variance ratio for each component
explained_variance_ratio = pca_3d.explained_variance_ratio_
print(f"Explained variance ratio (PC1, PC2, PC3): {explained_variance_ratio}")

# COMMAND ----------

import pandas as pd

cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X_pandas.columns)
print(cluster_centers_df)

# Printing summary statistics for each cluster
for cluster_num in range(n_clusters):
    print(f"\nCluster {cluster_num} - Summary Statistics:")
    cluster_data = X_scaled[y_kmeans == cluster_num]
    cluster_summary = pd.DataFrame(cluster_data).describe()
    print(cluster_summary)


# COMMAND ----------

def label_clusters(cluster_data):
    # High risk: Older bridges with high traffic and high precipitation
    if cluster_data['Bridge_Age_yr'] > 50 and cluster_data['Average_Daily_Traffic'] > 1000 and cluster_data['Total_Precipitation'] > 1000:
        return 'High Risk - Immediate Maintenance'
    
    # Moderate risk: Moderate age, moderate traffic, moderate precipitation
    elif 20 <= cluster_data['Bridge_Age_yr'] <= 50 and 500 <= cluster_data['Average_Daily_Traffic'] <= 1000 and cluster_data['Total_Precipitation'] <= 1000:
        return 'Moderate Risk - Regular Maintenance'
    
    # Low risk: Newer bridges with low traffic and low precipitation
    elif cluster_data['Bridge_Age_yr'] < 20 and cluster_data['Average_Daily_Traffic'] < 500 and cluster_data['Total_Precipitation'] < 500:
        return 'Low Risk - Low Maintenance'
    
    # Handle the edge cases (e.g., Cluster 3 with extreme conditions)
    elif cluster_data['Bridge_Age_yr'] < 20 and cluster_data['Average_Daily_Traffic'] < 500 and cluster_data['Total_Precipitation'] > 1000 and cluster_data['Average_Temperature'] > 25:
        return 'High Risk - Immediate Maintenance'
    else:
        return 'Unknown Risk'

cluster_labels = []
for i in range(len(X_pca_3d)):
    cluster_data = {
        'Bridge_Age_yr': X_pandas.iloc[i]['Bridge_Age_yr'],
        'Average_Daily_Traffic': X_pandas.iloc[i]['Average_Daily_Traffic'],
        'Total_Precipitation': X_pandas.iloc[i]['Total_Precipitation'],
        'Average_Temperature': X_pandas.iloc[i]['Average_Temperature']
    }
    label = label_clusters(cluster_data)
    cluster_labels.append(label)

X_pandas['Cluster Label'] = cluster_labels

# Now we have a 'Cluster Label' column in our DataFrame with risk categories


# COMMAND ----------

# Display distinct rows based on the 'Cluster Label' column with all columns
X_pandas.drop_duplicates(subset=['Cluster Label']).reset_index(drop=True)


# COMMAND ----------

import pandas as pd

# Create a pandas DataFrame with the cluster summary
cluster_summary = {
    "Cluster": [0, 1, 2, 3],
    "Traffic Level": ["Low (~2,317)", "High (~23,291)", "Very High (~162,670)", "Moderate (~65,623)"],
    "Bridge Age (Years)": ["Very Old (~51)", "Moderately Old (~44)", "Very Old (~51)", "Moderately Old (~46)"],
    "Temperature (°C)": ["Moderate (~14)", "Slightly Warm (~15)", "Slightly Warm (~15)", "Slightly Warm (~15)"],
    "Precipitation (mm)": ["Moderate (~1147)", "Low (~1131)", "Low (~1131)", "Very Low (~1130)"],
    "Risk Level": ["Moderate", "High", "Very High", "Low"]
}

df_cluster_summary = pd.DataFrame(cluster_summary)

# Display the DataFrame
display(df_cluster_summary)


# COMMAND ----------

## How does the condition of bridges differ between coastal and inland regions within a state, and what environmental factors contribute to these differences?

# COMMAND ----------

# Grouping by Location_Type and Bridge_Condition to count occurrences
condition_counts = df_cleaned.groupBy("Location_Type", "Bridge_Condition").count()

condition_counts.show()


# COMMAND ----------

from pyspark.sql.functions import col, count, avg

# Grouping by Location_Type and Bridge_Condition
condition_counts = df_cleaned.groupBy("Location_Type", "Bridge_Condition").count()

# Calculating total bridges per Location_Type
total_bridges = df_cleaned.groupBy("Location_Type").count().withColumnRenamed("count", "Total_Bridges")

# Calculate percentages
condition_percentages = condition_counts.join(total_bridges, "Location_Type") \
    .withColumn("Percentage", (col("count") / col("Total_Bridges")) * 100)

# Display results
condition_percentages.select("Location_Type", "Bridge_Condition", "Percentage").show()


# COMMAND ----------

# Group by Location_Type and calculate environmental averages
environmental_factors = df_cleaned.groupBy("Location_Type").agg(
    avg("Average_Temperature").alias("Avg_Temperature"),
    avg("Total_Precipitation").alias("Avg_Precipitation"),
    avg("Number_of_Days_with_Temperature_Below_0C").alias("Avg_Freezing_Days"),
    avg("Mean_Wind_Speed").alias("Avg_Wind_Speed")
)

# Display results
environmental_factors.show()


# COMMAND ----------

# Convert environmental factors to Pandas
environmental_factors_pandas = environmental_factors.toPandas()

# Plot bar charts for each environmental factor
factors = ["Avg_Temperature", "Avg_Precipitation", "Avg_Freezing_Days", "Avg_Wind_Speed"]
for factor in factors:
    plt.figure(figsize=(8, 5))
    plt.bar(environmental_factors_pandas["Location_Type"], environmental_factors_pandas[factor], color=["skyblue", "lightgreen"])
    plt.title(f"{factor.replace('_', ' ')} by Location Type")
    plt.xlabel("Location Type")
    plt.ylabel(factor.replace("_", " "))
    plt.grid(axis="y")
    plt.show()


# COMMAND ----------

from pyspark.sql.functions import when, col, lit, avg

# Group by Location_Type and Bridge_Condition
condition_counts = df_cleaned.groupBy("Location_Type", "Bridge_Condition").count()

# Calculate total bridges per Location_Type
total_bridges = df_cleaned.groupBy("Location_Type").count().withColumnRenamed("count", "Total_Bridges")

# Calculate percentages
condition_percentages = condition_counts.join(total_bridges, "Location_Type") \
    .withColumn("Percentage", (col("count") / col("Total_Bridges")) * 100)

# Separate calculations for Good, Fair, and Poor
condition_grouped = condition_percentages \
    .withColumn("Good_Percentage", when(col("Bridge_Condition") == "Good", col("Percentage")).otherwise(lit(None))) \
    .withColumn("Fair_Percentage", when(col("Bridge_Condition") == "Fair", col("Percentage")).otherwise(lit(None))) \
    .withColumn("Poor_Percentage", when(col("Bridge_Condition") == "Poor", col("Percentage")).otherwise(lit(None)))

# Aggregate by Location_Type for the final grouped output
final_grouped = condition_grouped.groupBy("Location_Type").agg(
    avg("Good_Percentage").alias("Good_Percentage"),
    avg("Fair_Percentage").alias("Fair_Percentage"),
    avg("Poor_Percentage").alias("Poor_Percentage")
)

# Display results
final_grouped.show()



# COMMAND ----------

from pyspark.sql.functions import col, when, avg

# Converting Bridge_Condition to a numeric scale: Poor = 0, Fair = 1, Good = 2
df_cleaned_numeric = df_cleaned.withColumn(
    "Bridge_Condition_Numeric",
    when(col("Bridge_Condition") == "Poor", 0)
    .when(col("Bridge_Condition") == "Fair", 1)
    .when(col("Bridge_Condition") == "Good", 2)
    .otherwise(None)
)

average_condition = df_cleaned_numeric.groupBy("Location_Type").agg(
    avg("Bridge_Condition_Numeric").alias("Average_Condition")
)

average_condition.show()

average_condition_pandas = average_condition.toPandas()

# Visualization: Bar plot for average bridge condition
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(
    x="Location_Type",
    y="Average_Condition",
    data=average_condition_pandas,
    palette="viridis"
)

plt.title("Average Bridge Condition by Location Type", fontsize=16)
plt.xlabel("Location Type", fontsize=12)
plt.ylabel("Average Condition (Numeric Scale)", fontsize=12)
plt.grid(axis="y")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Inland Data:
# MAGIC Good = 33.26%
# MAGIC Fair = 63.16%
# MAGIC Poor = 3.57%
# MAGIC Numeric Average for Inland:
# MAGIC
# MAGIC Avg_Condition
# MAGIC =
# MAGIC (
# MAGIC 2
# MAGIC ×
# MAGIC 33.26
# MAGIC )
# MAGIC +
# MAGIC (
# MAGIC 1
# MAGIC ×
# MAGIC 63.16
# MAGIC )
# MAGIC +
# MAGIC (
# MAGIC 0
# MAGIC ×
# MAGIC 3.57
# MAGIC )
# MAGIC  
# MAGIC /
# MAGIC  
# MAGIC 100
# MAGIC =
# MAGIC 66.52
# MAGIC +
# MAGIC 63.16
# MAGIC 100
# MAGIC =
# MAGIC 1.297
# MAGIC Avg_Condition=(2×33.26)+(1×63.16)+(0×3.57)/100= 
# MAGIC 100
# MAGIC 66.52+63.16
# MAGIC ​
# MAGIC  =1.297
# MAGIC
# MAGIC
# MAGIC Coastal Data:
# MAGIC Good = 31.71%
# MAGIC Fair = 65.23%
# MAGIC Poor = 3.07%
# MAGIC
# MAGIC
# MAGIC Numeric Average for Coastal:
# MAGIC
# MAGIC Avg_Condition
# MAGIC =
# MAGIC (
# MAGIC 2
# MAGIC ×
# MAGIC 31.71
# MAGIC )
# MAGIC +
# MAGIC (
# MAGIC 1
# MAGIC ×
# MAGIC 65.23
# MAGIC )
# MAGIC +
# MAGIC (
# MAGIC 0
# MAGIC ×
# MAGIC 3.07
# MAGIC )
# MAGIC  
# MAGIC /
# MAGIC  
# MAGIC 100
# MAGIC =
# MAGIC 63.42
# MAGIC +
# MAGIC 65.23
# MAGIC 100
# MAGIC =
# MAGIC 1.286
# MAGIC Avg_Condition=(2×31.71)+(1×65.23)+(0×3.07)/100= 
# MAGIC 100
# MAGIC 63.42+65.23
# MAGIC ​
# MAGIC  =1.286
# MAGIC  
# MAGIC Cross-Check Results:
# MAGIC
# MAGIC Coastal Calculated Average: 1.286
# MAGIC Matches with the value shown in the bar chart (slightly below Inland).
# MAGIC
# MAGIC Inland Calculated Average: 1.297
# MAGIC Matches with the value shown in the bar chart (slightly above Coastal).

# COMMAND ----------

correlation_data = df_cleaned.select(
    "Average_Relative_Humidity",
    "Average_Temperature", 
    "Number_of_Days_with_Measurable_Precipitation",
    "Number_of_Days_with_Temperature_Below_0C",
    "Total_Precipitation",
    "Average_Daily_Traffic",
    "Bridge_Age_yr" 
)

# Converting to Pandas for correlation analysis
correlation_data_pandas = correlation_data.toPandas()

# Computing Pearson correlation matrix
correlation_matrix = correlation_data_pandas.corr(method="pearson")

# Visualization: Heatmap for correlations
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Between Environmental Factors and Bridge Characteristics")
plt.show()


# COMMAND ----------

from pyspark.sql.functions import avg, col
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Handle null values in Bridge_Age_yr
df_cleaned = df_cleaned.filter(col("Bridge_Age_yr").isNotNull())

# Step 2: Group by Location_Type and calculate the average bridge age
bridge_age_stats = df_cleaned.groupBy("Location_Type").agg(
    avg("Bridge_Age_yr").alias("Avg_Bridge_Age")
).toPandas()

# Step 3: Extract Coastal and Inland bridge ages for statistical testing
coastal_ages = df_cleaned.filter(col("Location_Type") == "Coastal").select("Bridge_Age_yr").rdd.flatMap(lambda x: x).collect()
inland_ages = df_cleaned.filter(col("Location_Type") == "Inland").select("Bridge_Age_yr").rdd.flatMap(lambda x: x).collect()

# Perform a t-test to compare bridge ages between Coastal and Inland regions
t_stat, p_value = ttest_ind(coastal_ages, inland_ages, equal_var=False)

# Display the t-test results
print(f"T-Statistic: {t_stat:.2f}, P-Value: {p_value:.4f}")

# Step 4: Box plot for bridge age distribution across regions
bridge_age_pandas = df_cleaned.select("Location_Type", "Bridge_Age_yr").toPandas()

plt.figure(figsize=(8, 6))
sns.boxplot(x="Location_Type", y="Bridge_Age_yr", data=bridge_age_pandas, palette="coolwarm")
plt.title("Distribution of Bridge Age by Location Type", fontsize=16)
plt.xlabel("Location Type", fontsize=12)
plt.ylabel("Bridge Age (Years)", fontsize=12)
plt.grid(axis="y")
plt.show()

# Step 5: Analyze environmental factors' influence on bridge lifespan in Coastal vs. Inland regions
environmental_stats = df_cleaned.groupBy("Location_Type").agg(
    avg("Average_Temperature").alias("Avg_Temperature"),
    avg("Total_Precipitation").alias("Avg_Precipitation"),
    avg("Bridge_Age_yr").alias("Avg_Bridge_Age")
).toPandas()

# Step 6: Visualize environmental influence on bridge age
plt.figure(figsize=(10, 6))
sns.barplot(x="Location_Type", y="Avg_Temperature", data=environmental_stats, palette="viridis")
plt.title("Average Temperature by Location Type", fontsize=16)
plt.xlabel("Location Type", fontsize=12)
plt.ylabel("Average Temperature (°C)", fontsize=12)
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Location_Type", y="Avg_Precipitation", data=environmental_stats, palette="viridis")
plt.title("Average Precipitation by Location Type", fontsize=16)
plt.xlabel("Location Type", fontsize=12)
plt.ylabel("Average Precipitation (mm)", fontsize=12)
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="Location_Type", y="Avg_Bridge_Age", data=environmental_stats, palette="viridis")
plt.title("Average Bridge Age by Location Type", fontsize=16)
plt.xlabel("Location Type", fontsize=12)
plt.ylabel("Average Bridge Age (Years)", fontsize=12)
plt.grid(axis="y")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 1. T-Statistic and P-Value:
# MAGIC T-Statistic: -25.90
# MAGIC Indicates a significant difference between the average age of bridges in Coastal and Inland regions.
# MAGIC The negative value suggests that Coastal bridges tend to be younger compared to Inland bridges.
# MAGIC P-Value: 0.0000
# MAGIC Strongly significant, meaning the difference in bridge ages between Coastal and Inland regions is not due to random chance.
# MAGIC 2. Box Plot: Distribution of Bridge Age by Location
# MAGIC Coastal Bridges:
# MAGIC Tend to be younger with a smaller interquartile range (IQR), indicating less variation in age.
# MAGIC Outliers suggest a few older bridges exist, but most are relatively new.
# MAGIC Inland Bridges:
# MAGIC Tend to be older with a wider IQR, indicating more variability in age.
# MAGIC This aligns with the t-test results showing a significant difference in average age.
# MAGIC 3. Bar Plot: Average Temperature by Location
# MAGIC Coastal Regions:
# MAGIC Have higher average temperatures compared to Inland regions, likely due to proximity to water bodies that moderate temperature fluctuations.
# MAGIC Inland Regions:
# MAGIC Experience lower average temperatures, which may contribute to different structural challenges.
# MAGIC 4. Bar Plot: Average Precipitation by Location
# MAGIC Coastal Regions:
# MAGIC Have slightly higher average precipitation, possibly due to oceanic influences.
# MAGIC Inland Regions:
# MAGIC Experience less precipitation but may face challenges like snow or ice.
# MAGIC 5. Bar Plot: Average Bridge Age by Location
# MAGIC Coastal Bridges:
# MAGIC On average, they are significantly younger, which could indicate newer infrastructure investments in coastal areas to combat environmental stress (e.g., saltwater corrosion, hurricanes).
# MAGIC Inland Bridges:
# MAGIC Older bridges on average, which may reflect delayed infrastructure upgrades or historical development patterns.
# MAGIC

# COMMAND ----------

!pip install xgboost


# COMMAND ----------

from pyspark.sql import functions as F

df_cleaned = df_cleaned.withColumn('Bridge_Condition', F.trim(F.col('Bridge_Condition')))
df_cleaned.printSchema()

# Ensuring the 'Bridge_Condition' column is of type string
df_cleaned = df_cleaned.withColumn('Bridge_Condition', F.col('Bridge_Condition').cast('string'))

# Checking distinct values again after casting
df_cleaned.select("Bridge_Condition").distinct().show(truncate=False)



# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Processing and training the model with exception handling
def train_bridge_model(df_cleaned):
    try:
        # Step 1: Ensuring 'Bridge_Condition' exists and display its distinct values
        print("Displaying distinct values for 'Bridge_Condition' before transformation:")
        df_cleaned.select("Bridge_Condition").distinct().show()

        # Step 2: We assume 'Bridge_Condition' is already numeric (0, 1, 2 for Poor, Fair, Good)
        # Step 3: Cast numerical columns to double and handle missing/infinity values
        numerical_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']
        for col in numerical_columns:
            df_cleaned = df_cleaned.withColumn(col, F.col(col).cast(DoubleType()))
        
        for col in numerical_columns:
            # Computing the mean excluding null, NaN, or infinity values
            mean_value = df_cleaned.filter(
                (F.col(col).isNotNull()) & 
                (~F.col(col).isin(float('inf'), float('-inf')))
            ).agg(F.mean(F.col(col))).first()[0]

            # Replacing NaN, infinity, and extreme values
            df_cleaned = df_cleaned.fillna({col: mean_value})
            df_cleaned = df_cleaned.withColumn(
                col,
                F.when(F.col(col) == float('inf'), mean_value)
                 .when(F.col(col) == float('-inf'), mean_value)
                 .when(F.col(col) > 1e6, mean_value) 
                 .otherwise(F.col(col))
            )

        # Step 4: Preparing the features and target
        feature_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']
        X = df_cleaned.select(feature_columns)
        y = df_cleaned.select('Bridge_Condition')

        # Converting PySpark DataFrame
        X_pandas = X.toPandas()
        y_pandas = y.toPandas().squeeze()  # Convert DataFrame to Series

        # Ensuring no NaN or infinity in Pandas DataFrames
        X_pandas = X_pandas.replace([float('inf'), float('-inf')], None).fillna(X_pandas.mean())
        y_pandas = y_pandas.replace([float('inf'), float('-inf')], None).fillna(y_pandas.mode()[0])

        # Step 5: Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_pandas, y_pandas, test_size=0.2, random_state=42)

        # Step 6: Defining base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(random_state=42)),
            ('nn', MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42))
        ]

        # Step 7: Defining meta-model (logistic regression)
        meta_model = LogisticRegression(random_state=42)

        # Step 8: Creating the StackingClassifier ensemble model
        ensemble_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

        # Step 9: Training the ensemble model
        ensemble_model.fit(X_train, y_train)

        # Step 10: Evaluating the model
        y_pred = ensemble_model.predict(X_test)

        # Step 11: Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"An error occurred: {e}")

train_bridge_model(df_cleaned)


# COMMAND ----------

# MAGIC %md
# MAGIC As the above model is unable to give precision,recall and f1-score for 'Poor'  , we are using the SMOTE to enhance the model functionality of the Ensemble model

# COMMAND ----------

#%pip install imbalanced-learn


# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import SMOTE  
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

def train_bridge_model(df_cleaned):
    try:
        # Step 1: Ensuring 'Bridge_Condition' exists
        print("Displaying distinct values for 'Bridge_Condition' before transformation:")
        df_cleaned.select("Bridge_Condition").distinct().show()

        # Step 2: Numerical columns to double and handle missing/infinity values
        numerical_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']
        for col in numerical_columns:
            df_cleaned = df_cleaned.withColumn(col, F.col(col).cast(DoubleType()))
        
        # Replacing NaN, infinity, or large values with the column mean
        for col in numerical_columns:
            # Computing the mean excluding null, NaN, or infinity values
            mean_value = df_cleaned.filter(
                (F.col(col).isNotNull()) & 
                (~F.col(col).isin(float('inf'), float('-inf')))
            ).agg(F.mean(F.col(col))).first()[0]

    
            df_cleaned = df_cleaned.fillna({col: mean_value})
            df_cleaned = df_cleaned.withColumn(
                col,
                F.when(F.col(col) == float('inf'), mean_value)
                 .when(F.col(col) == float('-inf'), mean_value)
                 .when(F.col(col) > 1e6, mean_value)  
                 .otherwise(F.col(col))
            )

        # Step 3: Preparing the features and target
        feature_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']
        X = df_cleaned.select(feature_columns)
        y = df_cleaned.select('Bridge_Condition')

    
        X_pandas = X.toPandas()
        y_pandas = y.toPandas().squeeze()  # Converting DataFrame to Series

        
        X_pandas = X_pandas.replace([float('inf'), float('-inf')], None).fillna(X_pandas.mean())
        y_pandas = y_pandas.replace([float('inf'), float('-inf')], None).fillna(y_pandas.mode()[0])

        # Step 4: Resampling to handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_pandas, y_pandas)

        # Step 5: Feature Scaling with StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)

        # Step 6: Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

        # Step 7: Defining base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(random_state=42)),
            ('nn', MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=2000, random_state=42))  # Increase max_iter
        ]

        # Step 8: Define meta-model (logistic regression)
        meta_model = LogisticRegression(random_state=42, max_iter=2000)  # Increase max_iter

        # Step 9: Creating the StackingClassifier ensemble model
        ensemble_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

        # Step 10: Training the ensemble model
        ensemble_model.fit(X_train, y_train)

        # Step 11: Evaluating the model
        y_pred = ensemble_model.predict(X_test)

        # Step 12: Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        print(f"An error occurred: {e}")


train_bridge_model(df_cleaned)


# COMMAND ----------

from sklearn.model_selection import train_test_split
import pandas as pd

feature_columns = ['Bridge_Age_yr', 'Average_Daily_Traffic', 'Average_Temperature', 'Total_Precipitation']
target_column = 'Bridge_Condition'

# Step 1: Selecting features and target 
X = df_cleaned.select(feature_columns)  # Features
y = df_cleaned.select(target_column)    # Target variable

# Step 2:
X_pandas = X.toPandas()
y_pandas = y.toPandas().squeeze()  # Convert DataFrame to Series for y

# Ensuring no NaN or infinity in Pandas DataFrames
X_pandas = X_pandas.replace([float('inf'), float('-inf')], None).fillna(X_pandas.mean())
y_pandas = y_pandas.replace([float('inf'), float('-inf')], None).fillna(y_pandas.mode()[0])

# Step 3: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pandas, y_pandas, test_size=0.2, random_state=42)

# Checking the split and shape
print(X_train.shape, X_test.shape)  
print(y_train.shape, y_test.shape)  


# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'class_weight': ['balanced', None],
    'min_samples_split': [2, 5, 10],  # Number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Number of samples required to be at a leaf node
}

# Initializing Random Forest and GridSearchCV
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
rf_grid_search.fit(X_train, y_train)

print("Best Random Forest Params:", rf_grid_search.best_params_)
print("Best Random Forest Score:", rf_grid_search.best_score_)


# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encoding the target labels (y_train and y_test)
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Now we can proceed with XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# XGBoost hyperparameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'scale_pos_weight': [1, 2, 5],  # For handling imbalance
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples to train on
    'colsample_bytree': [0.8, 1.0],  # Fraction of features to use
}

# Initialize XGBoost and GridSearchCV
xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=3, n_jobs=-1, verbose=2)

# Fitting the model using encoded labels
xgb_grid_search.fit(X_train, y_train_encoded)

# Parameters and score for XGBoost
print("Best XGBoost Params:", xgb_grid_search.best_params_)
print("Best XGBoost Score:", xgb_grid_search.best_score_)


# COMMAND ----------

from sklearn.neural_network import MLPClassifier

# MLP Classifier hyperparameter grid
mlp_param_grid = {
    'hidden_layer_sizes': [(100,), (100, 100), (150, 100)],  # Number of neurons in each hidden layer
    'max_iter': [1000, 2000],
    'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
    'solver': ['adam', 'sgd'],  # Optimization method
}

# Initialize MLPClassifier and GridSearchCV
mlp_grid_search = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid, cv=3, n_jobs=-1, verbose=2)

# Fitting the model
mlp_grid_search.fit(X_train, y_train_encoded)

# Parameters and score for MLP Classifier
print("Best MLP Params:", mlp_grid_search.best_params_)
print("Best MLP Score:", mlp_grid_search.best_score_)


# COMMAND ----------

#%pip install imbalanced-learn


# COMMAND ----------

# MAGIC %md
# MAGIC We used Decision Tree with SMOTE to see the the differences and compare it

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Manually set the order to ensure "Poor" = 0, "Fair" = 1, "Good" = 2
label_encoder.classes_ = ['Poor', 'Fair', 'Good']

# Encode the target labels (y_train and y_test)
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Verify the encoding
print(f"Encoded classes: {label_encoder.classes_}")


# COMMAND ----------

from pyspark.sql import SparkSession
from imblearn.over_sampling import SMOTE

# Convert PySpark DataFrame to pandas DataFrame
y_pandas = y.toPandas()

# Convert to pandas Series or NumPy array
y_series = y_pandas.squeeze()  # Single-column DataFrame to Series
y_array = y_series.to_numpy()  # Series to NumPy array

# Check the shape of y_array
print(f"Shape of y_array: {y_array.shape}")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode target labels
y_encoded = label_encoder.fit_transform(y_array)

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pandas, y_encoded)

# Verify resampled shapes
print(f"X_resampled shape: {X_resampled.shape}")
print(f"y_resampled shape: {y_resampled.shape}")


# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Check shapes of the splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Retrain the Decision Tree model with the updated SMOTE data
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = decision_tree.predict(X_test)

# Print the classification report
print("Classification Report (Decision Tree with SMOTE):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm_decision_tree = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])  # Assuming 0=Poor, 1=Fair, 2=Good
plt.figure(figsize=(8, 6))
sns.heatmap(cm_decision_tree, annot=True, fmt="d", cmap="Blues", xticklabels=["Poor", "Fair", "Good"], yticklabels=["Poor", "Fair", "Good"])
plt.title("Confusion Matrix (Decision Tree with SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

