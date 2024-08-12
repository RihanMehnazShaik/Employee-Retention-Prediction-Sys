1.Server Setup
The server is built using Flask, a lightweight web framework for Python. Flask routes handle the different functionalities of the application, including training the model and making predictions.
Imports and Initialization:
•
Imports necessary libraries and modules, including Flask, pandas, and custom modules for training and prediction.
•
Initializes Flask application and Flask Monitoring Dashboard.
•
Enables Cross-Origin Resource Sharing (CORS).
Index Route (/):
•
Renders the index.html page for the base URL.
Training Route (/training):
•
Handles POST requests to start model training.
•
Uses a configuration object to get the run ID and data path.
•
Initializes a TrainModel object and calls its training_model method.
•
Returns a success message with the run ID or an error message if an exception occurs.
Single Prediction Route (/prediction):
•
Handles POST requests for single predictions.
•
Collects form data from the request, converts it to a DataFrame, and ensures correct data types.
•
Initializes a PredictModel object and calls its single_predict_from_model method.
•
Returns the predicted output or an error message if an exception occurs.
Batch Prediction Route (/batchprediction):
•
Handles POST requests for batch predictions.
•
Uses a configuration object to get the run ID and data path.
•
Initializes a PredictModel object and calls its batch_predict_from_model method.
•
Returns a success message with the run ID or an error message if an exception occurs.
Main Function:
•
Runs the application on host 0.0.0.0 and port 5000 using a WSGI server.


2.Dependencies and Environment
Project dependencies are listed in a requirements file, which allows for easy installation of all necessary packages. Using a virtual environment is recommended to manage these dependencies and avoid conflicts with other projects. This ensures that the project can be set up and run consistently across different environments.
3.Configuration Management
Manages file operations related to saving, loading, and selecting models.
save_model:
•
Purpose: Saves a model to a file.
•
Steps:
1.
Logs the start of the saving process.
2.
Constructs a path for the model file.
3.
Removes any existing directory for the models and creates a new one.
4.
Saves the model as a .sav file using pickle.
5.
Logs the successful save and returns 'success'.
6.
Exception Handling: Logs and raises exceptions.
load_model:
•
Purpose: Loads a model from a file.
•
Steps:
1.
Logs the start of the loading process.
2.
Opens the specified model file and loads it using pickle.
3.
Logs the successful load and returns the model.
4.
Exception Handling: Logs and raises exceptions.
correct_model:
•
Purpose: Finds the best model for a given cluster number.
•
Steps:
1.
Logs the start of the process.
2.
Lists files in the models directory.
3.
Searches for files that include the cluster number in their name.
4.
Extracts the model name and returns it.
4.Logging
Logging is integrated throughout the application to track events and errors. This helps in debugging and monitoring the application's performance. Important events, such as the start and end of training, parameter tuning results, and any errors encountered, are logged with appropriate severity levels.
5.Loading training dataset
This dataset contains employee records from a company, which can be used to analyze factors affecting employee turnover. Each row represents an individual employee, and each column provides specific information about the employee's characteristics and employment details.
Features:
•
empid: Unique identifier for each employee.
•
satisfaction_level: Employee's satisfaction level, ranging from 0 to 1.
•
last_evaluation: Last performance evaluation score, ranging from 0 to 1.
•
number_project: Number of projects the employee has been involved in.
•
average_montly_hours: Average monthly working hours.
•
time_spend_company: Number of years the employee has been with the company.
•
Work_accident: Indicator of whether the employee has had a work accident (0 = No, 1 = Yes).
•
promotion_last_5years: Indicator of whether the employee has been promoted in the last 5 years (0 = No, 1 = Yes).
•
salary: Salary level of the employee, categorized as 'low', 'medium', or 'high'.
•
left: Indicator of whether the employee has left the company (1 = Yes, 0 = No)
5.Validating the training dataset
•
Start Logging: The method starts by logging the beginning of the data load, validation, and transformation process.
•
Archive Old Files: It archives old files to prevent conflicts or overwriting during the data processing.
•
Extract Schema Information: The method extracts expected column names and the number of columns from a predefined schema (schema_train).
•
Validate Column Length: It checks whether the number of columns in the dataset matches the expected number of columns from the schema.
•
Validate Missing Values: The method checks if any column has all values missing and handles it appropriately.
•
Replace Missing Values: Blanks in the CSV file are replaced with "Null" values to standardize the data.
•
Create Database Table: The method creates a database table named training_raw_data_t under the training database using the extracted column names.
•
Insert Data into Table: It inserts the validated and transformed data from the CSV files into the database table.
•
Export Data to CSV: The data from the database table is exported back to a CSV file for further analysis or use.
•
Move Processed Files: Processed files are moved to a designated directory to keep track of processed and unprocessed files.
•
End Logging: Finally, the method logs the successful completion of the data load, validation, and transformation process.
•
Exception Handling: Any exceptions that occur during the process are caught, logged, and then re-raised to ensure proper debugging and error handling.
6.Preprocessing
Preprocessor class is designed to preprocess training and prediction datasets.
•
get_data():
1.
Reads the dataset from the specified path.
2.
Logs the start and end of the reading process.
•
drop_columns(data, columns):
1.
Drops specified columns from the dataset.
2.
Logs the start and end of the column dropping process.
•
is_null_present(data):
1.
Checks for missing values in the dataset.
2.
If missing values are found, logs the columns with missing values into a separate CSV file.
3.
Logs the start and end of the missing value detection process.
•
impute_missing_values(data):
1.
Uses KNNImputer to impute missing values in the dataset.
2.
Logs the start and end of the missing value imputation process.
•
feature_encoding(data):
1.
Encodes categorical features using dummy encoding.
2.
Logs the start and end of the feature encoding process.
•
split_features_label(data, label_name):
1.
Splits the dataset into features and labels.
2.
Logs the start and end of the splitting process.
•
preprocess_trainset():
Orchestrates the entire preprocessing pipeline:
1.
Reads data.
2.
Drops unwanted columns.
3.
Encodes categorical features.
4.
Checks for missing values and imputes if present.
5.
Splits data into features and labels.
6.
Logs the start and end of the preprocessing.
7.Clustering
KMeansCluster class is designed to perform KMeans clustering on a dataset.
•
elbow_plot(data):
1.
Plots the elbow method graph to determine the optimum number of clusters.
2.
Saves the plot locally.
3.
Determines the optimum number of clusters programmatically using the KneeLocator.
4.
Logs the start and end of the elbow plotting process.
•
create_clusters(data, number_of_clusters):
1.
Performs KMeans clustering on the dataset with the specified number of clusters.
2.
Saves the trained KMeans model.
3.
Adds a new column to the dataset indicating the cluster to which each data point belongs.
4.
Logs the start and end of the cluster creation process.
This class provides methods to perform KMeans clustering on a dataset, including determining the optimal number of clusters using the elbow method and creating clusters based on the specified number.
8.Model Tuning
ModelTuner class is responsible for tuning and selecting the best machine learning model from a given set of hyperparameters.
Initialization:
Upon instantiation, the class initializes essential components such as logger and initializes instances of the RandomForestClassifier and XGBClassifier from scikit-learn and XGBoost libraries, respectively.
best_params_randomforest() Method:
This method finds the best parameters for the Random Forest algorithm using grid search with cross-validation.
It initializes a grid search object (GridSearchCV) with a predefined set of hyperparameters for the Random Forest algorithm.
Grid search is performed over the parameter grid, and the best parameters are identified based on the highest performance metric (accuracy or AUC).
A new Random Forest model is instantiated with the best parameters and trained on the training data.
best_params_xgboost() Method:
Similar to best_params_randomforest(), this method finds the best parameters for the XGBoost algorithm using grid search with cross-validation.
It initializes a grid search object with a predefined set of hyperparameters for the XGBoost algorithm and performs grid search to identify the best parameters.
A new XGBoost model is instantiated with the best parameters and trained on the training data.
get_best_model() Method:
This method compares the performance of the tuned Random Forest and XGBoost models using the provided training and test data.
It computes the accuracy or AUC score for both models based on the test data.
The model with the higher performance metric is selected as the best model.
The method returns the name of the best model ('RandomForest' or 'XGBoost') along with the corresponding model object.
9.Training Process
Model Selection :Two machine learning models are used in this project: XGBoost and Random Forest Classifier.
XGBoost:
XGBoost, or eXtreme Gradient Boosting, is a machine learning algorithm under ensemble learning. It is trendy for supervised learning tasks, such as regression and classification. XGBoost builds a predictive model by combining the predictions of multiple individual models, often decision trees, in an iterative manner.
The algorithm works by sequentially adding weak learners to the ensemble, with each new learner focusing on correcting the errors made by the existing ones. It uses a gradient descent optimization technique to minimize a predefined loss function during training.
Parameters in XGBoost:
1.
Learning Rate (eta): An important variable that modifies how much each tree contributes to the final prediction. While more trees are needed, smaller values frequently result in more accurate models.
2.
Max Depth: This parameter controls the depth of every tree, avoiding overfitting and being essential to controlling the model’s complexity.
3.
Gamma: Based on the decrease in loss, it determines when a node in the tree will split. The algorithm becomes more conservative with a higher gamma value, avoiding splits that don’t appreciably lower the loss. It aids in managing tree complexity.
4.
Subsample: Manages the percentage of data that is sampled at random to grow each tree, hence lowering variance and enhancing generalization. Setting it too low, though, could result in underfitting.
5.
Colsample Bytree: Establishes the percentage of features that will be sampled at random for growing each tree.
6.
n_estimators: Specifies the number of boosting rounds.
7.
lambda (L2 regularization term) and alpha (L1 regularization term): Control the strength of L2 and L1 regularization, respectively. A higher value results in stronger regularization.
8.
min_child_weight: Influences the tree structure by controlling the minimum amount of data required to create a new node.
9.
scale_pos_weight: Useful in imbalanced class scenarios to control the balance of positive and negative weights.
Why XGboost?
XGBoost is highly scalable and efficient as It is designed to handle large datasets with millions or even billions of instances and features.
XGBoost implements parallel processing techniques and utilizes hardware optimization, such as GPU acceleration, to speed up the training process. This scalability and efficiency make XGBoost suitable for big data applications and real-time predictions.
It provides a wide range of customizable parameters and regularization techniques, allowing users to fine-tune the model according to their specific needs.
Random Forest Classifier:
Random Forest Classification is an ensemble learning technique designed to enhance the accuracy and robustness of classification tasks. The algorithm builds a multitude of decision trees during training and outputs the class that is the mode of the classification classes. Each decision tree in the random forest is constructed using a subset of the training data and a random subset of features introducing diversity among the trees, making the model more robust and less prone to overfitting.
The random forest algorithm employs a technique called bagging (Bootstrap Aggregating) to create these diverse subsets.
Parameters in Random Forest Classifier:
1.
n_estimators: Number of trees in the forest.
More trees generally lead to better performance, but at the cost of computational time.
Start with a value of 100 and increase as needed.
2.
max_depth: Maximum depth of each tree.
Deeper trees can capture more complex patterns, but also risk overfitting.
Experiment with values between 5 and 15, and consider lower values for smaller datasets.
3.
max_features: Number of features considered for splitting at each node.
A common value is ‘sqrt’ (square root of the total number of features).
Adjust based on dataset size and feature importance.
4.
criterion: Function used to measure split quality (‘gini’ or ‘entropy’).
Gini impurity is often slightly faster, but both are generally similar in performance.
5.
min_samples_split: Minimum samples required to split a node.
Higher values can prevent overfitting, but too high can hinder model complexity.
Start with 2 and adjust as needed.
6.
min_samples_leaf: Minimum samples required to be at a leaf node.
Similar to min_samples_split, but focused on leaf nodes.
Start with 1 and adjust as needed.
7.
bootstrap: Whether to use bootstrap sampling when building trees (True or False).
Bootstrapping can improve model variance and generalization, but can slightly increase bias.
Why Random Forest Classifier?
The Random Forest Classifier is a robust machine learning algorithm known for its high accuracy and versatility. It constructs multiple decision trees during training and combines their predictions to yield a final result, making it resistant to overfitting. This ensemble approach enables Random Forests to handle large datasets efficiently and remain robust to noisy data and outliers.
An advantage of the Random Forest Classifier is its ability to provide feature importance scores, aiding in feature selection and interpretation. Additionally, it requires minimal hyperparameter tuning and is straightforward to implement, making it accessible to both novice and experienced practitioners. Its parallelizable nature allows for faster training times, contributing to its widespread use in various domains.
TrainModel class: It is responsible for training machine learning models using a modular and organized approach.
It begins by logging the start of the training process and noting the run ID for tracking purposes.
The method then proceeds with data loading, validation, and transformation using the LoadValidate class instance.
Next, it performs preprocessing activities such as handling missing values, encoding categorical features, and saving column information to a JSON file.
The method utilizes the KMeansCluster class to create clusters based on the data and determines the optimal number of clusters using an elbow plot.
Data is divided into clusters, and for each cluster:
Features and labels are prepared.
The data is split into training and test sets.
The ModelTuner class is used to identify the best machine learning model for the cluster based on performance metrics.
The best model is saved to the directory using the FileOperation class.
9.Prediction Process
For making predictions, the server receives input data from the client-side form.
Loading and Validating the prediction dataset: Data validation ensures that the prediction dataset is clean and suitable for prediction process.
Preprocessing: Preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical features.
The data undergoes the same preprocessing steps as during training to ensure consistency.The preprocessed data is fed into the trained models to generate predictions. The server then sends the prediction result back to the client, where it is displayed to the user.
10.File Operations
File operations are essential for saving and loading the trained models. The implementation includes methods for saving a trained model to a file and loading it back when needed. This ensures that the model does not need to be retrained every time the application is restarted, saving time and computational resources.
11.User Interface
The user interface is designed to be intuitive and user-friendly. The main interface is an HTML form where users can input various employee details such as satisfaction level, last evaluation, number of projects, average monthly hours, time spent at the company, work accidents, promotions in the last five years, and salary. This form is styled with Bootstrap to ensure a clean and responsive design.
The form submission is handled by JavaScript, which captures the input data and sends it to the server via AJAX. This allows the prediction to be made without reloading the page, providing a seamless user experience. Upon receiving the prediction result from the server, the JavaScript code updates the UI to display the result dynamically.
12.Client-Side Scripting
Client-side scripting is implemented using JavaScript and jQuery. When the user submits the form, JavaScript prevents the default form submission behavior and instead sends an AJAX request to the server. This request includes the employee details entered by
the user.The AJAX request is configured to send the data in the appropriate format and handle the server's response. When the server returns the prediction result, the client-side script updates the UI to display this result, giving immediate feedback to the user
