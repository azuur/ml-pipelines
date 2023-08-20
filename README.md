# ml-pipelines
Sanbdox to play with ML pipelines and the cloud

## Changelog

- 2033/08/19: First commits
- 2023/08/20: Added training script

## Backlog

- ~~Write ML training script~~
- Write model evaluation script
- Figure out how to run Kedro/Airflow/etc in AWS EC2 or Lambdas
- Package pipelines with Kedro, Airflow, or something else
- Figure out how to set up simple database + data lineage in AWS
- Figure out how to set up model registry in AWS
- Add and package data collection pipeline from AWS database
- Write dockerfiles to run data collection pipeline, train model and evaluate, serve model
- Figure out how to use GitHub Actions to:
    - Build and run data collection + model training + model eval pipeline
    - Generating PR with new model metrics and id
    - Upon merging automated PR, build and serve model
- Read about building Docker images with poetry + other stuff related to project structure
    - https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
    - https://towardsdatascience.com/automating-version-tags-and-changelogs-for-your-python-projects-6c46b68c7139
    - https://towardsdatascience.com/setting-up-python-projects-part-v-206df3c1e3d3
