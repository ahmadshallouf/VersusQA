# VersusQA
## Introduction

This is a repository for the VersusQA project. The goal of this project is to create a question answering system that can answer questions about the differences between two entities. For example, given the entities "Harry Potter" and "LotR", the system is able to answer a question such as "What is better: Harry Potter or LotR?". The arguments for "Harry Potter" and "LotR" are displayed along with a clear and coherent summary.

## Sub-Reposities

This repository contains the following sub-repositories:

- [CQI](./CQI): Contains the code for the Comparative Questions Identification (CQI) task.
- [OAI](./OAI): Contains the code for the Objects and Aspects Identification (OAI) task.
- [SC](./SC): Contains the code for the Stance Classification (SC) task.
- [CQAS](./CQAS): Contains the code for the Comparative Question Answering Summarization (CQAS) task.
- [backend](./backend): Contains the code for the backend of the VersusQA system.
- [frontend](./frontend): Contains the code for the frontend of the VersusQA system.

## Local Setup
### With Docker
1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
2. Clone this repository.
3. Run `docker-compose up` in the [./backend](./backend) directory of this repository.
4. Frontend is available at [http://localhost:15557](http://localhost:15557).
5. Backend is available at [http://localhost:15558](http://localhost:15558).

### Without Docker
1. Install [Python 3.8](https://www.python.org/downloads/release/python-380/) and [pip](https://pip.pypa.io/en/stable/installing/).
2. Clone this repository.
3. Create a virtual environment with `python3 -m venv venv` in all sub-repositories.
4. Activate the virtual environment with `source venv/bin/activate` in all sub-repositories (separate terminals).
5. Install the dependencies with `pip install -r requirements.txt` in all sub-repositories.
6. In each sub-repository, run the FastAPI server with `uvicorn main:app --host=0.0.0.0 --port=[See Below] --reload`.
    - [CQI](./CQI): `--port=8001`
    - [OAI](./OAI): `--port=8002`
    - [SC](./SC): `--port=8003`
    - [CQAS](./CQAS): `--port=8004`
7. Run a PostgresSQL database server using an installation on your local machine or a Docker container.
    - If you use a Docker container, run `docker run --name postgres -p 5432:5432 -d postgres` in a terminal.
    - If you use an installation on your local machine, run `sudo service postgresql start` in a terminal.
8. Change the database connection string, the username and password in the [./backend/src/main/resources/application.properties](./backend/src/main/resources/application.properties) file.
9. Install [Maven](https://maven.apache.org/install.html).
10. Build the Spring Boot server with `mvn clean install` in the [./backend](./backend) directory.
11. Run the Spring Boot server with `java -jar target/Comparative-Question-Answering---Backend-0.0.1-SNAPSHOT.jar` in the [./backend](./backend) directory.
12. Navigate to the [./frontend](./frontend) directory.
13. Install [Node.js](https://nodejs.org/en/download/) and [npm](https://www.npmjs.com/get-npm).
14. Install the Angular CLI with `npm install -g @angular/cli`.
15. Install the dependencies with `npm install`.
16. Run the Angular frontend with `ng serve` in the [./frontend](./frontend) directory.
17. Frontend is available at [http://localhost:4200](http://localhost:4200).
18. Backend is available at [http://localhost:8080](http://localhost:8080).

## Deployment on a production server
### With Docker
1. Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
2. Clone this repository.
3. Change the backend URL in the [./frontend/src/environments/environment.prod.ts](./frontend/src/environments/environment.prod.ts) file to your exposed backend URL.
4. Run `docker-compose -f docker-compose.prod.yml up` in the [./backend](./backend) directory of this repository.
5. Map the ports 15557 and 15558 to your exposed ports, with backend on 15558 and frontend on 15557.

## Contributors
This System was built during a Master Project at the University of Hamburg. The following people contributed to this project:

- [Ahmad Shallouf](https://github.com/ahmadshallouf)
- [Hanna Herasimchyk](https://github.com/geranium12)
- [Rudy Garrido](https://github.com/rudygarrido)
- [Natia Mestvirishvili](https://github.com/nmestvirishvili)

This project was under the supervision of the Language Technology Group at the University of Hamburg.

## License
This project is licensed under the terms of the Apache 2.0 license. See [LICENSE](./LICENSE) for more information.

## Deployed System
The deployed system is available at [https://cam-v2.ltdemos.informatik.uni-hamburg.de/](https://cam-v2.ltdemos.informatik.uni-hamburg.de/).
