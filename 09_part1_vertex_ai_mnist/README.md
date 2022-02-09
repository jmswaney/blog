# Custom training jobs with Vertex AI on GCP

This post is Part 1 of a X part series. Part 1 covers writing our custom training job source code.

## Training package

GCP has you upload an entire Python package that encapsulates all of your training code. GCP will containerize you Pyhton package and execute it on their platform. This allows you to write your source code as you normally would with version control and your local development setup but have scalable power of cloud GPUs and TPUs available to you from Google's infrastructure. This allows you to train much bigger models and much faster than you could locally.

- It looks like pip can install poetry projects, so it may work with train jobs
