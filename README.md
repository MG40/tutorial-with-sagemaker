# Running XGBoost as a Container Job on AWS SageMaker

Running XGBoost as a container instead of a SageMaker notebook instance can significantly reduce costs. With containers, you only pay for the compute resources used during the training and inference processes, without the need to keep an instance running for exploratory work. This is especially beneficial for:

- **Batch jobs**: Execute training on-demand without maintaining idle resources.
- **Scaling**: Use multiple containers for parallel processing only when necessary.
- **Flexibility**: Integrate training directly into CI/CD pipelines or other workflows.

This approach minimizes long-term operational expenses while offering a scalable, efficient, and cost-effective alternative to traditional notebook instances.

## Prerequisites
- Install the AWS SageMaker SDK by running:
  ```bash
  pip install sagemaker
  ```
- Ensure your AWS CLI is configured with the correct credentials and region.
- Install any additional dependencies, such as `shap` for data preparation.

## Step 1: Data Preparation

### Load and Analyze the Dataset
The dataset is loaded and analyzed using the SHAP library:
```python
import shap
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)
```

### Split the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Save the Data in CSV Format
```python
train.to_csv('train.csv', index=False, header=False)
validation.to_csv('validation.csv', index=False, header=False)
```

## Step 2: Upload Data to S3
```python
import boto3
bucket = sagemaker.Session().default_bucket()
prefix = "demo-sagemaker-xgboost-adult-income-prediction"

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/train.csv')
).upload_file('train.csv')

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/validation.csv')
).upload_file('validation.csv')
```

## Step 3: Configure SageMaker Training
### Specify the Container Image
```python
from sagemaker import image_uris
region = sagemaker.Session().boto_region_name
container = image_uris.retrieve("xgboost", region, "1.2-1")
```

### Define the Estimator
```python
from sagemaker.estimator import Estimator
xgb_model = Estimator(
    image_uri=container,
    role="arn:aws:iam::<your_account_id>:role/<your_role>",
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=f"s3://{bucket}/{prefix}/xgboost_model",
    sagemaker_session=sagemaker.Session()
)

xgb_model.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
    objective="binary:logistic",
    num_round=1000
)
```

## Step 4: Train the Model
```python
from sagemaker.session import TrainingInput

train_input = TrainingInput(f"s3://{bucket}/{prefix}/data/train.csv", content_type="csv")
validation_input = TrainingInput(f"s3://{bucket}/{prefix}/data/validation.csv", content_type="csv")

xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)
```

## Step 5: Deploy the Model
```python
from sagemaker.serializers import CSVSerializer
xgb_predictor = xgb_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    serializer=CSVSerializer()
)
```

## Step 6: Cost Benefits of Using a Container
Running XGBoost as a container instead of a SageMaker notebook instance can significantly reduce costs. With containers, you only pay for the compute resources used during the training and inference processes, without the need to keep an instance running for exploratory work. This is especially beneficial for:

- **Batch jobs**: Execute training on-demand without maintaining idle resources.
- **Scaling**: Use multiple containers for parallel processing only when necessary.
- **Flexibility**: Integrate training directly into CI/CD pipelines or other workflows.

This approach minimizes long-term operational expenses while offering a scalable, efficient, and cost-effective alternative to traditional notebook instances.

## Step 7: Cleanup Resources
```bash
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```
