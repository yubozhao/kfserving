# Predict on an InferenceService using BentoML

The goal of custom image support is to allow users to bring their own wrapped model
inside a container and serve it with KFServing. Please note that you will need to
ensure that your container is also running a web server e.g. Flask to expose your model
endpoints. This example located in the `model-server` directory extends
`kfserving.KFModel` which uses the tornado web server.

This example

[BentoML](https://bentoml.org) is an open-source framework for high-performance ML model serving.

This example will build a classifier model using iris dataset with BentoML, build and
push docker image to Docker Hub, and then deploy it to a cluster with KFServing installed.


## Deploy a custom InferenceService using BentoML

Install Jupyter and the other dependencies needed to run the python notebook

```shell
pip install -r requirements.txt
```

Start Jupyter and open the notebook

```shell
jupyter notebook custom-bentoml.ipynb
```

Follow the instructions in the notebook to deploy the InferenseService.
