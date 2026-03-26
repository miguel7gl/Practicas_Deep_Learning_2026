# Deep Learning Lab Sessions

Welcome to the Deep Learning Lab Sessions! In these sessions, we will explore various concepts and applications of deep learning. This readme file will provide you with an overview of each session, its objectives, and what you can expect to learn.

By the end of these sessions, you will have a solid understanding of convolutional filters, transfer learning, and how to deploy a deep learning model in a real-world application.

Happy learning!

## Session 1: Understanding Convolutional Filters

**Objective:** Gain a comprehensive understanding of convolutional filters and their role in Convolutional Neural Networks (CNNs).

In this session, we will dive into the workings of convolutional filters, which are the building blocks of Convolutional Neural Networks (CNNs). We will discuss how these filters help in extracting features from images, and how they contribute to the success of CNNs in computer vision tasks and how can we try to make CNNs more interpretable.

## Sessions 2-4: Transfer Learning and Deployment

**Objective:** Learn how to apply transfer learning to a simulated business case and deploy the model in a publicly available [Streamlit](https://streamlit.io/) app using [FastAPI](https://fastapi.tiangolo.com/) as back-end.

In these sessions, we will focus on applying transfer learning to a real-world simulated business case. We will make use of several pre-trained model and fine-tune it to suit our specific needs. Once our model is ready, we will deploy it in a publicly available Streamlit app, allowing us to test the results in real life. We will discuss on best practices for Deep Learning experimentation,
which can be costly and time consuming and we will make use of [Weights & Biases](https://wandb.ai/) web app.

## Final project deliverable
**Objective of the project** - the deliverable must demonstrate that you are able to:

1. Select an appropriate pre-trained neural network and apply transfer learning to the real-estate image classification use case.
2. Run professional-grade experimentation and hyperparameter tuning with [Weights & Biases](https://wandb.ai/) in a team workflow.
3. Deploy a production-ready inference service using [Streamlit](https://streamlit.io/) (front-end) and [FastAPI](https://fastapi.tiangolo.com/) (back-end API).

### Required final deliverable

The final deliverable is a **technical report** supported by a public repository and runnable deployment artifacts. It must include

- **Customer context (real-estate marketplace):** Business problem, target user, expected value, and operational constraints.
- **System architecture:** High-level architecture of the solution.
- **Modeling approach:** Selected pre-trained model, transfer-learning strategy, and final neural network architecture used by the API.
- **Experimentation process (W&B):** Experiment design, hyperparameter search strategy, tracked runs, and final model selection criteria.
- **Performance metrics per output class:** Expected quality level for customer usage, for example class-wise precision, recall, F1-score and confusion matrix interpretation.
- **API documentation:** OpenAPI/Swagger documentation and endpoint behavior (inputs, outputs, and error handling).
- **Project links:** Link to the Git repository and link to the W&B project/workspace.
- **Access requirements:** Invite `agascon@comillas.edu` and `rkramer@comillas.edu` to your W&B project/workspace, and make the Git repository public.

To be considered complete, the submission must include:

- A reproducible codebase with setup instructions.
- A working API with accessible Swagger docs.
- A working Streamlit app connected to the API.
- A traceable W&B experimentation history.
- A final report with conclusions and business-facing recommendations.

### Evaluation rubric

- **60% Experimentation quality:** Model selection rationale (including pre-trained NN choice), transfer-learning strategy, hyperparameter tuning depth, and quality/reproducibility of W&B tracking.
- **20% Deployment quality:** Robustness and usability of the FastAPI + Streamlit production setup, including API documentation and end-to-end operability.
- **20% Report clarity and business relevance:** Technical clarity, decision justification, and relevance of conclusions for the real-estate marketplace customer.

### Report format constraints

- Maximum length: **6 pages**.
- **No cover page**.
