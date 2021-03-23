# Disaster Response Pipeline for Figure Eight
This project is part of the Udacity - Data Science Nano-Degree.

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is to apply data engineering skills, to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Methods Used
* ETL Pipeline
* ML Pipeline
* Flask Web App

### Technologies (Currently)
* Python
* Jupyter, Pandas, Numpy
* SQLite
* scikit-learn

## Project Description
In this project, I applied data engineering skills to analyse disaster data from Figure Eight to build a model for an API that classifies disaster messages. I created an Extract, Transform and load (ETL) process that takes in the messages and categories datasets and merges the two datasets, cleans the data and stores it in a SQLite database. I then created a machine learning pipeline to categorise these events so that I could send the messages to an appropriate disaster relief agency. My project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualisations of the data.

## Needs of this project
(To be completed)

## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).



## 6. Project Organisation

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │  
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │   └── train_model.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   └── train_classifier.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
          └── visualize.py
