# Assignment 1 - Andres Garcia

## Downloading and Installing Dependencies

1. Download .zip file
2. Unzip contents into your MA-ML4NLP-LABS/code directory
3. Then follow the steps below in the terminal:
4. Commands are in Linux CDI but should be very similar in other OS's

```
# If you dont have virtualenv installed run:

python3 -m pip install virtualenv

# Create and activate an environment:

python3 -m virtualenv .venv
source .venv/bin/activate

# Go to my submission folder and install dependencies:

cd code/assignment2
pip install -r requirements.txt
```

## Running the Code
You can use both the jupyter or the python file version. The jupyter should just run by running the notebook using our environemnt as kernel. For the python version, navigate to my submission folder and run:

```
python3 basic_system.py [-h] [--evaluation {token,span}] [--mode {default,hyperparam}] [train_file] [dev_file] [test_file] [results_path] [embeddings_file]
```

#### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `dev_file`        | Path to the development file         |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

#### **Options**
| Option                         | Description                                                    |
|--------------------------------|----------------------------------------------------------------|
| `-h, --help`                   | Show this help message and exit                                |
| `--evaluation {token,span}`    | Evaluation type: `token` or `span`                            |
| `--mode {default,hyperparam}`  | Mode of operation: train 3 models (`default`) or hyperparameter tuning (`hyperparam`) |


