# 1. Use an official Python runtime as a parent image
# Python 3.12-slim is chosen for its balance between features and image size
FROM python:3.12-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy only the requirements file first 
# This leverages Docker's layer caching for faster subsequent builds
COPY requirements.txt .

# 4. Install production dependencies
# --no-cache-dir keeps the image slim by not storing the index of installed packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the local code and model artifacts to the container
# This includes main.py and loan_model.joblib
COPY . .

# 6. Set environment variables
# Cloud Run automatically sets the PORT environment variable to 8080
ENV PORT 8080
ENV PYTHONUNBUFFERED True

# 7. Run the web service on container startup 
# Using uvicorn to serve the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]