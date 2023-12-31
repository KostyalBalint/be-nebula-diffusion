# Use an official Python runtime as a parent image
FROM anibali/pytorch:2.0.1-nocuda

# Set the working directory in the container
WORKDIR /app

# Copy all files from the current directory to the container
COPY . /app

# Install requirements.txt
RUN pip install -r requirements.txt

# Run the Python application
CMD ["python", "main.py"]