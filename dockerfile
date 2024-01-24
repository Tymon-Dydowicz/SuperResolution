FROM python:slim

WORKDIR /data
COPY ./ISR/raw_data/ /data/ISR
COPY ./Own /data/Own

WORKDIR /models
COPY ./models /models

WORKDIR /metrics
COPY ./metrics /metrics

WORKDIR /images
COPY ./images /images

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make ports 7860 and 6006 available to the world outside this container
EXPOSE 7860
EXPOSE 6006

# Run main.py when the container launches
CMD ["python", "main.py"]