#Create a API of Dataframe
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("LR").getOrCreate() 
df=spark.read.csv("path",inferSchema=True,header=True)

#To show the number of rows and columns of dataset
print("The number of rows are {} and the numbr of colums are {}".format(df.count(),len(df.columns)))

#To get a schema of dataset
df.printSchema() 

#To select a desire column and get vital information about its statistical measures by describe function.
df.select('Name of column').describe().show()

#To calaculate statistical measures about whole dataset 
df.describe().show()

#To calculate correlation between columns
from pyspark.sql.functions import  corr
df.select(corr('Name of column','target')).show()

#To Calculate the correlation between all features and target
for col in df.columns :
    print("The correlation between {} and target is : {}".format(col, df.corr(col,'output')) )
    
#We create a single vector combining all input features by using Spark’s VectorAssembler. It creates only a single feature that captures the input values for that row.
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler 
features_df=df_vector.transform(df)
features_df.printSchema()

#Select the features column and target
model_df=features_df.select('features','target')
model_df.show()

#We have to split the dataset into a training and test dataset
train_df,test_df=model_df.randomSplit([0.7,0.3])
print(train_df.count(),test_df.count())

#we build and train the Linear Regression model using features of the input and target columns.
from pyspark.ml.regression import LinearRegression
Lin_reg=LinearRegression(labelCol='target')
lr_model=Lin_reg.fit(train_df)
lr_model.set

#Coefficients
print(lr_model.coefficients)

#Intercept
print(lr_model.intercept)

#Evaluate the performance of model on training data as well using r2
train_prediction=lr_model.evaluate(train_df)
print(train_prediction.r2)

#Evaluate the performance of model on test data as well using r2
test_predictions=lr_model.evaluate(test_df)
print(test_predictions.r2)


#Show the Target and predicted value by model
predictions = lr_model.transform(test_df)
predictions.select("prediction","Target","features").show()
