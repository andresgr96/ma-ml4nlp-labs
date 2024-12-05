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

cd code/assignment3
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

### Running Feature Ablation Analysis
To run the feature ablation analysis, you can use the feature_ablation.py script provided. 

### Example Usage
```
python3 feature_ablation.py [-h] [--model {SVM,logreg,NB}] [train_file] [dev_file] [test_file] [results_path] [embeddings_file]
```

### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `dev_file`        | Path to the development file         |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

### **Options**
| Option                  | Description                                                   |
|-------------------------|---------------------------------------------------------------|
| `-h, --help`            | Show this help message and exit                               |
| `--model {SVM,logreg,NB}` | Specify the model type for the feature ablation analysis. Options include `SVM`, `logreg`, or `NB`. |

### Running Error Analysis
To perform error analysis on the NER system, you can use the error_anal.py script provided. 

### Example Usage
To run the error analysis script:

```
python3 error_anal.py [-h] [train_file] [test_file] [results_path] [embeddings_file]
```

### **Positional Arguments**
| Argument          | Description                          |
|-------------------|--------------------------------------|
| `train_file`      | Path to the training file            |
| `test_file`       | Path to the test file                |
| `results_path`    | Directory to save results            |
| `embeddings_file` | Path to the Word2Vec embeddings file |

### **Options**
| Option       | Description                              |
|--------------|------------------------------------------|
| `-h, --help` | Show this help message and exit          |

### Running Error Visualization
To visualize errors from the NER system, use the provided Python script. Navigate to the appropriate directory and execute the script as follows:

```
error_vis.py [-h] [--focus_classes FOCUS_CLASSES [FOCUS_CLASSES ...]] [error_file] [output_dir]

```


### **Positional Arguments**
| Argument      | Description                                     |
|---------------|-------------------------------------------------|
| `error_file`  | Path to the prediction analysis CSV file        |
| `output_dir`  | Directory to save results                      |

### **Options**
| Option                                 | Description                                                                |
|----------------------------------------|----------------------------------------------------------------------------|
| `-h, --help`                           | Show this help message and exit                                            |
| `--focus_classes FOCUS_CLASSES [FOCUS_CLASSES ...]` | List of classes to focus on for error analysis (default: `['I-LOC', 'I-MISC', 'I-ORG']`) |




