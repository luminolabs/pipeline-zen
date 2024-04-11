FROM ubuntu:latest

COPY ../common ./project
COPY src/* ./project

ENTRYPOINT ["python", "main.py", "some_job_id"]