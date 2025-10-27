# Start from the prepared base image
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Install FastAPI, Uvicorn, and scikit-learn
RUN pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn

# Copy FastAPI app into the container
COPY main.py .

# Expose port 8000 for the API
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
