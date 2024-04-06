# Start from the official Python 3.9 image optimized for ARM64 architecture
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Install the required packages
RUN pip install --upgrade pip && \
    python3 -m pip install tensorflow && \
    pip install pandas numpy tqdm pyarrow transformers tf-keras

# Copy the local directory contents into the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "train.py"]
