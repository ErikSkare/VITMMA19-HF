# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Erik Skáre (Z7ZF6D)
- **Aiming for +1 Mark**: Yes

### Data preparation

Created a script (00-data-preparation.py) which automatically downloads the raw dataset and transforms it into the appropriate format.

It runs automatically before the rest of the pipeline. As a result I didn't mount the data folder, because the data gets downloaded automatically inside the container.

### Solution Description

The goal of this project is to classify paragraphs of legal texts based on understandableness (1-5). The dataset consists of a few thousand individually annotated legal paragraphs (used for training and validation), and hundred consensus-labeled paragraphs (better representation, used for testing).

The task can be interpreted as an ordinal regression problem. As a result I've chosen MAE (mean absolute error) as an evaluation metric (for validation and testings). It've found it much more representative for the task than accuracy, because it weights misclassifications.

I've cleaned the datasets by removing missing labels and data leakage from consensus to individual. This way the result of the evaluation isn't inflated and can be trusted.

As a baseline model I've developed a simple logistic regression with hand-crafted features such as: character count, word count, sentence count, average word length and punctuation count. I've also optimized L2 regularization parameter by estimating test-error with cross-validation for each configuration.

For the stronger model I developed an 1D CNN network on the token embeddings generated with HuSpaCy. I've used CORAL for the loss (better suited for ordinal regression). During optimization, I upweighted larger errors to penalize misclassifications with greater ordinal distance more heavily than adjacent ones. To handle varying paragraph lengths, I applied batch-wise padding and used adaptive max pooling.

The final model has about 21k parameters, all trainable.

### Extra Credit Justification

Compared cross-entropy with CORAL loss, turned out the latter was a better suited inductive bias for the problem. This way I achieved better generalization. Additionally, weighting was applied to decrease large misclassifications. Small misclassifications do not significantly impact the model’s usefulness, but larges do.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command.

```bash
docker run dl-project > log/run.log 2>&1
```

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `00-data-preparation.py`: Downloads raw dataset and transforms it into appropriate format.
    - `01-data-preprocessing.py`: Cleans and preprocesses individual and consensus datasets.
    - `02-training.py`: Trains the baseline, and final model (1D CNN).
    - `03-evaluation.py`: Evaluates the baseline, and final model on the consensus dataset.
    - `04-inference.py`: Runs the baseline, and final model on new, unseen data.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - `models.py`: Defines the deep learning model (1D CNN).

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
