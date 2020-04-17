# Predict on an InferenceService using BentoML

In this example, it builds a classifier model using the iris data set with Scikit-learn
and save the model with BentoML, builds and pushes an API server image to the Docker Hub,
and then deploys the images as InferenceService to a cluster with KFServing installed for
inferencing.

[BentoML](https://bentoml.org) is an open-source framework for high-performance ML model
serving. It bundles ML model, preprocessing/post-processing code, model dependencies
and model configuration into an API server that is ready to deploy to platforms such as
AWS Lambda, Sagemaker, Google Cloud Run, Kubernetes, KNative, and more.

BentoML supports all major machine learning frameworks including Keras, Tensorflow, PyTorch, Fast.ai, XGBoost and etc.

## Deploy a custom InferenceService using BentoML

### Setup

* Your ~/.kube/config should point to a cluster with KFServing installed.
* Your cluster's Istio Ingress gateway must be network accessible.
* Docker and Docker hub must be properly configured
* Python 3.6 or above
  * Install required packages `bentoml` and `scikit-learn`:
    ```shell
    pip install bentoml scikit-learn
    ```

### Save classification model with BentoML

BentoML creates a model API server, via prediction service abstraction.

The following code defines a prediction service that requires a scikit-learn model,
and asks BentoML to figure out the required PyPI pip packages automatically. It
also defined an API, which is the entry point for accessing this prediction service.
And the API is expecting a `pandas.DataFrame` object as its input data.

Save the code to a new file named `iris_classifier.py`:

```python
from bentoml import env, artifacts, api, BentoService
from bentoml.handlers import DataframeHandler
from bentoml.artifact import SklearnModelArtifact

@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
class IrisClassifier(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.model.predict(df)
```

Run the following code to train a classifier model and save it with BentoML

```python
from sklearn import svm
from sklearn import datasets

from iris_classifier import IrisClassifier

if __name__ == "__main__":
    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)

    # Create a iris classifier service instance
    iris_classifier_service = IrisClassifier()

    # Pack the newly trained model artifact
    iris_classifier_service.pack('model', clf)

    # Save the prediction service to disk for model serving
    saved_path = iris_classifier_service.save()
```

### Deploy InferenceService

BentoML generates a Dockerfile for model API server during the model saving process. Use
that Dockerfile to build and push an API server image to Docker Hub.

```shell
# Replace DOCKER_USERNAME with the Docker Hub username
docker_username=DOCKER_USERNAME
model_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

docker build -t $docker_username/iris-classifier $model_path

docker push $docker_username/iris-classifier
```

BentoML's REST interface is different than the Tensorflow V1 HTTP API that KFServing
expects. Requests send directly to the prediction service and bypass the top-level
InferenceService.

*Note: Support for KFserving V2 prediction protocol with BentoML is coming soon.*


Replace the `{docker_username}` with your Docker Hub user name and save to a file named
`bentoml.yaml`:

```yaml
apiVersion: serving.kubeflow.org/v1alpha2
kind: InferenceService
metadata:
  labels:
    controller-tools.k8s.io: "1.0"
  name: iris-classifier
spec:
  default:
    predictor:
      custom:
        container:
          image: {docker_username}/iris-classifier
          ports:
            - containerPort: 5000
```

Use `kubectl apply` to create new InferenceService:

```shell
kubectl apply -f bentoml.yaml
```

### Run prediction

*Note: Use kfserving-ingressgateway as your INGRESS_GATEWAY if you are deploying
KFServing as part of Kubeflow install, and not independently.*

```shell
MODEL_NAME=iris-classifier
INGRESS_GATEWAY=istio-ingressgateway
CLUSTER_IP=$(kubectl -n istio-system get service $INGRESS_GATEWAY -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" \
  --header "Content-Type: application/json" \
  --request POST \
  --data '[[5.1, 3.5, 1.4, 0.2]]' \
  http://$CLUSTER_IP/model/predict
```

### Delete deployment

```shell
kubectl delete -f bentoml.yaml
```