FROM node:lts

RUN mkdir /usr/src/app

WORKDIR /usr/src/app

RUN npm install -g @angular/cli

COPY . .

ARG PROFILE
ENV PROFILE $PROFILE
RUN echo "Environment: ${PROFILE}"

EXPOSE 4200

ENTRYPOINT bash -c "npm install && if [[ "$PROFILE" == "production" ]] then ng serve --host 0.0.0.0 --port 4200 --configuration=production --disable-host-check; elif [[ "$PROFILE" == "docker" ]] then ng serve --host 0.0.0.0 --port 4200 --configuration=docker --disable-host-check; ng serve --host 0.0.0.0 --port 4200 --disable-host-check; fi"
