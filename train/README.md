### How to run the train workflow

- cd to this folder
- `docker build -t train-workflow .`
- `docker run -v "$PWD/src/.cache":/project/.cache -v "$PWD/src/.results":/project/.results train-workflow`