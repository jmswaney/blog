# Custom training jobs with Vertex AI on GCP

This post is Part 2 of a X part series. Part 2 covers model training, evaluation, and deployment.

## Environment setup

- Create and link a billing account (if you don't have one)
  - Put a link to pricing and a cost estimate for this tutorial
- Enable the Vertex AI API
- We don't need to create a dataset because we will be using MNIST, which is built-in with Keras
  - You will need a bucket though for our source code
- Train new model
  - Training method
    - No need for a managed dataset or labels
    - Custom training selected
  - Model details
    - Name `mnist_mobilenetv2`
    - No advanced details
  - Training container
    - Pre-built container
    - TF 2.1 (default)
    - Select our source code

## Training

Run the model training briefly

## Evaluation

Check validation accuracy or F1-score

## Deployment

Deploy to an endpoint and make predictions. Try to use a pre-built container for this so that we don't need to do another custom packaging.
