# Disaster Response Pipeline Project
This app inputs messages and codes them based on 36 categories to assist with prioritization of disaster relief efforts.

The front end is a web app that allows users to check new messages and to see a few visualizations from the training dataset.
### Table of Contents

1. [Instructions](#Instructions)
2. [File Descriptions](#files)
3. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## File Descriptions <a name="files"></a>
HTML files of web app front end:

go.HTML
master.HTML

Python files:
run.py
process_data.py
train_classifier.py

Data:
disaster_categories.csv
disaster_messages.csv


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
I would like to thank Figure Eight for providing the labeled data used in the training of the ML model. As well as Udacity for the course work outlining this project.
