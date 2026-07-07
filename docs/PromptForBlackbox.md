You are a Senior Machine Learning Engineer.

Project Name
Apartment Rent Prediction using Artificial Neural Network

Programming Language
Python

IDE
Visual Studio Code

Architecture
Modular / Clean Architecture

Coding Style
PEP8

Objective
Build a complete ANN model to predict apartment rental prices and provide a production-ready ML application.

Requirements
Use TensorFlow
Use Pandas
Use NumPy
Use sklearn
Use Matplotlib
Use Streamlit for UI

Write reusable code.

Every module must have comments.
Every function must have docstrings.
Never duplicate code.
Split preprocessing into separate modules.
Separate training and evaluation.
Separate UI from Business Logic.
Separate Business Logic from Data Layer.
Use service layer.
Use config layer for all settings/hyperparameters.
Use utility layer for shared helpers.

Always save model.
Always save scaler.
Always save encoder.
Generate charts automatically.
Log every training process.
Generate prediction CSV.

Production rules
- Python type hints
- Exception handling: never allow unhandled crash
- Use logging in every module
- Every path uses pathlib (no hard-coded absolute paths)
- Never hard-code hyperparameters (must come from config)
- Always validate user input
- Prefer dependency injection when appropriate
- Optimize for maintainability and scalability

Extra deliverables
- docs/AI_Assistant.md: explain metrics in Vietnamese for non-ML users
- Streamlit pages: Dataset Explorer, Dashboard, Model Manager, Prediction History, Settings

