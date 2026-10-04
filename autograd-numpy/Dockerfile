FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .
VOLUME ["/app/results", "/app/data"]
ENTRYPOINT ["python", "-m", "autograd_numpy.train"]
CMD ["--optimizer", "both", "--epochs", "10"]
