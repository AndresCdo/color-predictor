FROM node:22-alpine AS deps
WORKDIR /app
COPY app/package.json app/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY app/package.json app/next.config.mjs app/jsconfig.json app/.eslintrc.json ./
COPY app/src ./src
RUN npm test
RUN npm run build

FROM nginx:stable-alpine AS prod
COPY conf/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/out /usr/share/nginx/html/color-predictor

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:3000/color-predictor/" >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
