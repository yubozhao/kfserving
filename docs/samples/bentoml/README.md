# Predict on an InferenceService using BentoML

In this example, you will build a classifier model using iris data set with Scikit-learn
and save the model with BentoML, build and push a docker image to the Docker Hub, and then
deploy it as InferenceService to a cluster with KFServing installed for inferencing.

[BentoML](https://bentoml.org) is an open-source framework for high performance ML model
serving, which supports all major machine learning frameworks including Keras,
Tensorflow, PyTorch, Fast.ai, XGBoost and etc.

## Deploy a custom InferenceService using BentoML

Install Jupyter and the other dependencies needed to run the python notebook

```shell
pip install -r requirements.txt
```

Start Jupyter and open the notebook

```shell
jupyter notebook bentoml.ipynb
```

Follow the instructions in the notebook to deploy the InferenseService.
