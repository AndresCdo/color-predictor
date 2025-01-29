FROM node:18-alpine AS build

COPY app/ /app
WORKDIR /app

# Next.js build
RUN npm install
RUN npm run build --prod

FROM nginx:1.17.1-alpine AS prod
COPY --from=build /app/.next /usr/share/nginx/html
COPY --from=build /app/public /usr/share/nginx/html

COPY ./conf/nginx.conf /etc/nginx/nginx.conf
COPY ./conf/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
