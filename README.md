# Opinion Dynamics: The Bounded Confidence Model

This is a repository on the bounded confidence model of opinion dynamics, as first put forward by [Hegselmann & Krause](https://www.jasss.org/5/3/2.html) (2002). It also contains an implementation of the linear pooling model (see, for instance, ([Lehrer, 1976](https://doi.org/10.2307/2214612))). This repository contains code for the model and for some visualizations. 

[![A picture of an example of a run of a bounded confidence model](/images/example.png  "An example of a bounded confidence model")]()

## 1. Setup
To run the project, you first need to install the required packages
```commandline
pip install -r requirements.txt
```

## 2. The model `model.py`
The class `BoundedConfidence` in the module `model.py` is an implementation of the bounded confidence model. The class `LinearPooling` is an implementation of the linear pooling model.

## 3. Visualization `visualize.py`
The method `visualize` in the module `visualize.py` can be used to visualize the run of a bounded confidence model. 

## 4. Figures `figures.py`
The module produces all the images that are relevant to the paper. 

## A. Licence and citation
This repository accompanies ongoing research. In the meantime, please cite as follows:

- Duijf, H. (2024). An implementation and visualization of the bounded confidence model. https://github.com/HeinDuijf/BoundedConfidenceModel 

Released under the [MIT licence](LICENCE.md).
