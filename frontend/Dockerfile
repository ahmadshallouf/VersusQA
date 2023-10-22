FROM node:lts

RUN mkdir /usr/src/app

WORKDIR /usr/src/app

RUN npm install -g @angular/cli

COPY . .

EXPOSE 4200

ENTRYPOINT bash -c "npm install && ng serve --host 0.0.0.0 --port 4200"