# GNN Image Recognition Project

This project implements an image recognition algorithm based on Graph Neural Networks (GNN) using the MNIST dataset.

## Project Structure
project/
│
├── data/
│   ├── __init__.py
│   ├── load_data.py
│
├── models/
│   ├── __init__.py
│   ├── gcn.py
│
├── train.py
├── evaluate.py
├── optimize.py
├── predict.py
├── gui.py
│
├── requirements.txt
└── README.md


- `data/`: Contains data loading scripts.
- `models/`: Contains model definitions.
- `train.py`: Script for training the model.
- `evaluate.py`: Script for evaluating the model.
- `optimize.py`: Script for optimizing the model.
- `predict.py`: Script for using the model to make predictions.
- `gui.py`: GUI application for interacting with the model.

## Requirements

- `torch`
- `torch-geometric`
- `openai`
- `tk`

Install the requirements using:
- `pip install -r requirements.txt`


### Training

Use the GUI to train the model, either with your own dataset or with data generated using ChatGPT.

### Evaluation

Use the GUI to evaluate the trained model.

### Prediction

Use the GUI to make predictions using the trained model.


