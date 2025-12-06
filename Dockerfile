# Use a lightweight Python image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /app

# Install system dependencies (if needed later, we can add more)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better build caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose ports:
# 8501 for Streamlit
# 8000 for FastAPI (if you choose to run it in this container)
EXPOSE 8501
EXPOSE 8000

# Default command: run the Streamlit dashboard
# You can override this at runtime if you want to run FastAPI instead.
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
