<p align="center">
 <img src="https://github.com/ThugPigeon653/testQL-source/blob/0ed0945349c10f06f79b6d5ecdcbd45d560081ca/testQL-logo.PNG" alt="TestQL - SQL unit testing made easy"></a>
</p>

<h3 align="center">ML Fine-tuning with AWS</h3>

<div align="center">

</div>

---

<p align = "center">💡 Using Docker and AWS to fine tune a CNN (Convolutional Neural Network)</p>


## Table of Contents

- [About](#about)

See intro video: https://youtu.be/3Biv_cqomlk

## About <a name = "about"></a>

- The tensorlow model is initially trained using NIST data. The data is a set of handwritted numbers, which are presented to the NN as 28x28px single-channel images. Once the model has finished training on the initial data, it can be fine-tuned by feeding it batches of new data. 
- The client application focuses on being able to easily deliver images to s3, with correct labelling. A number will be displayed on the screen, and the user has to write it in their handwriting. It is then saved as an image of appropriate format, and uploaded to s3. 
- Once S3 contains enough images for a batch, a lambda launches the NN trainer in an EB environment. The trainer assesses whether the model has performed better, given the new data - if it has, the model is sent to another s3 bucket for delivery.
- A cloudformation template has been provided, for simple deployment to AWS services.
- The actual application uses Docker Compose, to decouple the NN trainer by business concern. This allows seperate teams to mantain different parts of the system. It also means the application is very easy to deploy.
- To run locally, just use 'docker compose up'

