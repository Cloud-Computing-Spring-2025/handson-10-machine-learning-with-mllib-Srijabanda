from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnAnalysis").getOrCreate()

# Load dataset
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)

# Prepare output file
output_file = "model_outputs.txt"
with open(output_file, "w") as f:
    f.write("Customer Churn Analysis Results\n")
    f.write("===============================\n\n")

# Task 1: Data Preprocessing and Feature Engineering
def preprocess(df):
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges").cast("double")))

    categorical = ["gender", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=c, outputCol=c + "Idx") for c in categorical]
    encoders = [OneHotEncoder(inputCol=c + "Idx", outputCol=c + "Vec") for c in categorical]
    
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    features = numeric + [c + "Vec" for c in categorical]

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    stages = indexers + encoders + [label_indexer, assembler]
    
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    processed = model.transform(df)

    with open(output_file, "a") as f:
        f.write("=== Data Preprocessing ===\n")
        f.write("Transformed sample (features and label):\n")
        for row in processed.select("features", "label").take(5):
            f.write(f"{row}\n")
        f.write("\n")
    return processed.select("features", "label")

# Task 2: Logistic Regression Evaluation
def evaluate_logistic_regression(data):
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    model = LogisticRegression()
    fitted = model.fit(train)
    predictions = fitted.transform(test)

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    with open(output_file, "a") as f:
        f.write("=== Logistic Regression ===\n")
        f.write(f"AUC Score: {auc:.4f}\n\n")

# Task 3: Chi-Square Feature Selection
def feature_selection(data):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    reduced = selector.fit(data).transform(data)

    with open(output_file, "a") as f:
        f.write("=== Chi-Square Feature Selection ===\n")
        f.write("Top 5 selected features (first 5 rows):\n")
        for row in reduced.select("selectedFeatures", "label").take(5):
            f.write(f"{row}\n")
        f.write("\n")

# Task 4: Hyperparameter Tuning and Model Comparison
def model_comparison(data):
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBTClassifier": GBTClassifier()
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [5, 10]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].maxDepth, [10, 15]).addGrid(models["RandomForest"].numTrees, [20, 50]).build(),
        "GBTClassifier": ParamGridBuilder().addGrid(models["GBTClassifier"].maxDepth, [5, 10]).addGrid(models["GBTClassifier"].maxIter, [10, 20]).build()
    }

    best_model = None
    best_model_name = ""
    highest_auc = 0.0

    with open(output_file, "a") as f:
        f.write("=== Model Comparison & Tuning ===\n")
        for name, model in models.items():
            f.write(f"Tuning {name}...\n")
            grid = param_grids[name]
            cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
            cv_model = cv.fit(train)
            auc = evaluator.evaluate(cv_model.transform(test))
            f.write(f"{name} AUC: {auc:.4f}\n")
            if auc > highest_auc:
                best_model = cv_model
                best_model_name = name
                highest_auc = auc
        f.write(f"Best Model: {best_model_name}, AUC = {highest_auc:.4f}\n")

# Execute pipeline
processed_df = preprocess(df)
evaluate_logistic_regression(processed_df)
feature_selection(processed_df)
model_comparison(processed_df)

# Stop Spark
spark.stop()
