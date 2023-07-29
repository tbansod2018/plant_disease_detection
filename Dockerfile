# Use the official Python image as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files to the container
COPY app.py /app
COPY disease_info.csv /app
COPY best_model64.h5 /app/best_model64.h5
COPY templates /app/templates

# Install required dependencies
RUN pip install flask tensorflow pandas pillow

# Expose the port that Flask app will be running on
EXPOSE 5000

# Define the command to start your Flask application
CMD ["python", "app.py"]
