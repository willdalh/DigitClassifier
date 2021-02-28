FROM node:lts as build
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY package.json /usr/src/app

RUN npm install

COPY . /usr/src/app
RUN npm run build

FROM nginx:1.14.2
COPY --from=build /usr/src/app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /etc/nginx/certs
COPY privkey.pem /etc/nginx/certs/privkey.pem
COPY fullchain.pem /etc/nginx/certs/fullchain.pem

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]