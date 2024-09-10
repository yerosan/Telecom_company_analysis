# Using an official Python runtime as a parent image
FROM python:3.8-slim

# Setting the working directory in the container
WORKDIR /app

# Copying the current directory contents into the container
COPY . .

# Installing the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Exposing the required port for running the API
EXPOSE 8000

# Command to run the API with Uvicorn
CMD ["uvicorn", "run_model_service:app", "--host", "0.0.0.0", "--port", "8000"]

