FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY environment.py .
COPY filesystem_gen.py .
COPY scenarios/ /orwd_data/scenarios/
COPY tasks/ /orwd_data/tasks/
COPY scenarios/compromised_laptop/base_tree.json /orwd_data/base_tree.json

ENV DECEPTIONSEARCH_DATA_DIR=/orwd_data
EXPOSE 8080

CMD ["python", "server.py"]
