FROM maven AS TEMP_BUILD_IMAGE
USER root

ENV APP_HOME=/usr/app/

WORKDIR $APP_HOME
COPY ./src $APP_HOME/src
COPY ./pom.xml $APP_HOME/pom.xml
RUN mvn install -DskipTests

FROM maven
ENV ARTIFACT_NAME=Comparative-Question-Answering---Backend-0.0.1-SNAPSHOT.jar
ENV APP_HOME=/usr/app/
ENV CQI_URL=http://cqi:8001/is_comparative
ENV OAI_URL=http://oai:8002/get_objects_and_aspects
ENV CAM_URL=http://ltdemos.informatik.uni-hamburg.de/cam-api
ENV CSI_URL=http://csi:8003/get_arguments
ENV CQAS_URL=http://cqas:8004/summary
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=admin
ENV DB_URL=jdbc:postgresql://db:5432/postgres

COPY --from=TEMP_BUILD_IMAGE $APP_HOME/target/$ARTIFACT_NAME .
ENTRYPOINT ["java","-jar","Comparative-Question-Answering---Backend-0.0.1-SNAPSHOT.jar"]
