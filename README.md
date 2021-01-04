# MLFlow for experiment tracking

[MLFlow](https://www.mlflow.org/) self describes as

> an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

While going all-in on MLFlow represents a substantial architectural decision, employing experiment tracking is a low friction way of improving a model research and development process.
This project demonstrates instrumenting some training scripts with MLFlow's tracking component to that end.

## Structure

The folder structure of this repo is as follows

```
.
├── cml       # This folder contains scripts that facilitate the project launch on CML.
└── scripts   # Our analysis code
```

### cml

These scripts are specific to Cloudera Machine Learning, and, with the `.project-metadata.yaml` file in the root directory, allow the project to be deployed automatically, following a declarative specification for jobs, model endpoints and applications.

```
cml
├── install_dependencies.py # Script to run pip install of Python dependencies
└── mlflow_ui.py            # Script to launch MLFlow ui application.
```

### scripts

This is where all our analysis code lives.
In a more involved analysis, we could replace these scripts with jupyter notebooks to run manually, or abstract some re-usable code into a Python libary.

```
scripts
├── data.py                 # create fake train and test data
├── train_kneighbors.py     # train a k-nearest neighbors classifier
└── train_random_forest.py  # train a random forest classifier
```

## Running through the analysis

If this repo is imported as an Applied Machine Learning Prototype, the launch process should handle all the setup for you.
In case you want to run through it manually, follow the Installation instructions below.

### Installation

The code was developed for Python 3.6.9, and will likely work on more later versions.
To install dependencies, first create and activate a new virtual environment through your preferred means, then pip install from the requirements file. For instance:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are working in CML or CDSW, no virtual environment is necessary. Instead, inside a Python 3 session, simply run

```
!pip3 install -r requirements.xt
```

### Scripts

Inside the `scripts/` directory are three scripts, as described above.
The `data.py` script creates a fake dataset for a classification problem.
When tacklinga real business problem, we'd probably be reading this data from a database or flat file storage.

We have instrumented two simple machine learning algorithms to predict our target variable from the features, and each has its own training script.
`train_kneighbors.py` trains a k-nearest neighbors algorithm, where the number of neighbors to consider is set at the top of the file.
`train_random_forest.py` trains a random forest, and we expose two hyperparameters&mdash;the maximum tree depth and number of trees&mdash;also at the top of the file.

To run an experiment, simply run either script in a CML session.
