# Disaster Response Pipeline for Figure Eight

## Project Description
In this project, I applied data engineering skills to analyse disaster data from Figure Eight to build a model for an API that classifies disaster messages. I created an Extract, Transform and load (ETL) process that takes in the messages and categories datasets and merges the two datasets, cleans the data and stores it in a SQLite database. I then created a machine learning pipeline to categorise these events so that I could send the messages to an appropriate disaster relief agency. My project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualisations of the data.


## 1. Disaster Response - Dataset
The data set contains 26,248 real messages that were sent during disaster events. These messages are classified as either 'direct', 'social' or 'news' and those are the labels we will use to train the nlp model.

When looking at the distribution of message by genre we can see the majority comes from the news:
<kbd> <img src="https://raw.githubusercontent.com/waltersmac/Disaster-Response-Pipelines/master/figures/genre_pie.png" alt="drawing"/> </kbd>

## 2. ETL Pipeline
Link for notebook is [here](https://github.com/waltersmac/Disaster-Response-Pipelines/blob/master/notebooks/ETL%20Pipeline%20Preparation.ipynb) <br/>

In the data directory, the 'process_data.py' file performs the data cleaning pipeline that:

  * Loads the messages and categories datasets
  * Merges the two datasets
  * Cleans the data
  * Stores it in a SQLite database

Note - Please run "make requirements" to ensure the required modules are installed.

To run ETL pipeline that cleans data and stores in database:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`


## 3. Machine Learning Text Classifier Pipeline
Link for notebook is [here](https://github.com/waltersmac/Disaster-Response-Pipelines/blob/master/notebooks/ML%20Pipeline%20Preparation.ipynb) <br/>

In the model directory, the 'train_classifier.py' file performs the machine learning pipeline that:

  * Loads data from the SQLite database
  * Splits the dataset into training and test sets
  * Builds a text processing and machine learning pipeline
  * Trains and tunes a model using a SearchCV approach
  * Outputs results on the test set
  * Exports the final model as a pickle file

To run ML pipeline that trains classifier and saves:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## 4. Model Evaluation
After researching various models I found that the best NLP pipeline and model tested was based on a Multioutput ClassifierChain classifier and a lightGBM tree algorithm. <br/>
<kbd> <img src="https://raw.githubusercontent.com/waltersmac/Disaster-Response-Pipelines/master/figures/classification_report.png" alt="drawing" style="width:550px;"/> </kbd>

## 5. Flask Web App
I amended a web app template from Udacity in which an emergency worker can input a new message and ten get classification results in several categories. The app also outputs a visual of how the machine learning model made the latest classification of the text entered by the user. In addition to flask, the web app template that also uses html, css, javascript and Plotly visualisations.

To start the web app locally, run the following command in the app directory and then go to http://0.0.0.0:3001 to interact with the app:
python app/run.py



## 6. Project Organisation

    ├── Makefile           <- Makefile with commands like `make requirements`
    ├── README.md          <- The top-level README for developers using this project.
    ├── app                <- Flask application scripts
    ├── data               <- Hold the messages and category csv files, also the process_data script
    ├── figures            <- Saved images from the notebooks and flask app webpage
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`


## Resources Used for This Project
* Udacity Data Science Nanodegree: [here](https://www.udacity.com/course/data-scientist-nanodegree--nd025) <br>
* lightgbm: LGBMClassifier documentation: [here](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#) <br>
