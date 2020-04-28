# Predict on an InferenceService using BentoML

This example deploys an iris classifier model API server built with BentoML as
an InferenceService to a KFServing installed cluster.

[BentoML](https://bentoml.org) is an open-source framework for high-performance ML model
serving. BentoML supports all major machine learning frameworks including Keras,
Tensorflow, PyTorch, Fast.ai, XGBoost and etc.

## Deploy a custom InferenceService

### Setup

* Your ~/.kube/config should point to a cluster with KFServing installed.
* Your cluster's Istio Ingress gateway must be network accessible.
* Docker and Docker hub must be properly configured on your local system
* Python 3.6 or above
  * Install required packages `bentoml` and `scikit-learn` on your local system:

    ```shell
    pip install bentoml scikit-learn
    ```

### Build API model server using BentoML

Save the following code to a file named `iris_classifier.py`:

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

This code defines a model API server that requires a `scikit-learn` model, and asks BentoML
to figure out the required PyPI pip packages automatically. It also defined an API,
which is the entry point for accessing this prediction service. And the API is expecting
a `pandas.DataFrame` object as its input data.

Run the following code to create a BentoService SavedBundle with the trained an iris
classification model. A BentoService SavedBundle is a versioned file archive ready for
production deployment. The archive contains the model server defined above, python code
dependencies and PyPi dependencies, and the trained iris classification model:

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

Find the file directory of the SavedBundle with `bentoml get` command, which is
directory structured as a docker build context. Running docker build with this
directory produces a docker image containing the model API server. Replace
`docker_username` with your Docker Hub username and run the following code:

```shell
# Replace DOCKER_USERNAME with the Docker Hub username
docker_username=DOCKER_USERNAME
model_path=$(bentoml get IrisClassifier:latest -q | jq -r ".uri.uri")

docker build -t $docker_username/iris-classifier $model_path

docker push $docker_username/iris-classifier
```

*Note: BentoML's REST interface is different than the Tensorflow V1 HTTP API that
KFServing expects. Requests will send directly to the prediction service and bypass the
top-level InferenceService.*

*Support for KFServing V2 prediction protocol with BentoML is coming soon.*

Replace the `{docker_username}` with your Docker Hub username and save the code to a
file named `bentoml.yaml`:

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

Use `kubectl apply` to create a new InferenceService:

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