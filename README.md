# tides-shared-services

Repo containing shared micro service containers for tides_pipe and tidestom

## Contents

- SNID API
- NGSF API

## Maintainence
If you are adding a new shared service to this repository you should create a new top level directory with the name of the shared service with the following stucture:
```
tides-shared-services
|  README.md
|  docker-compose.yml
|
|- snid_docker
|- nsgf_docker
|- new_service_docker
    Dockerfile
    requirements.txt
    new_service.py
```

## Development and Installation Instructions
### Deveopment
To develop on any of the shared services contained within this repository you should clone this repository, and create a new branch ```[your-user]-dev```, when you are ready to merge in your changes you should create a pull request to the ```dev``` branch. You should _never_ merge into ```main```.

### Installation
This contents of this repo should not be installed in isolation, as they rely on networked volumes which are defined in [tides-integrated-system](https://github.com/TiDES-4MOST/tides-integrated-system/tree/main), thus any attempt to build the containers from the ```docker-compose``` in this repository will lead to unexpected results.
