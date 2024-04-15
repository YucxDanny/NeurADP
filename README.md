# NeurADP for Ride-pooling

## Introduction
This is the unofficial implemented code in **PyTorch** for the paper "Neural Approximate Dynamic Programming for On-Demand Ride-Pooling" that appears in the Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence.

## Requirements

We recommend using python<=3.7 and a conda venv.

```
conda create --name NeurADP python=3.7
cd path/to/NeurADP
conda activate NeurADP
```

Install the dependencies with:

```
pip install -r requirements.txt
```
**Pay attention to the installation of `docplex`!**
If you only want to use the free version of `docplex`, simply:
```
pip install docplex
```
If you want to use the full version of `docplex`, you need to install it from source. (or contact the developer)
```
git clone https://github.com/IBMDecisionOptimization/docplex-docplex.git
cd docplex-docplex
python setup.py install
```
When you're done working on the project, deactivate the conda virtual environment with `deactivate`.

## Data

Here is the structure of the data folder:

```
data/
    files_60sec/
        test_flow_5000_1.txt
        test_flow_5000_2.txt
    ignorezonelist.txt
    taxi_3000_final.txt
    zone_path.csv
    zone_traveltime.csv
```

## Usage

To run the code, simply:

```
python main.py
```

The code will automatically process the data in the `data` folder.