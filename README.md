### Setup le docker

```bash
docker-compose up -d --build
```

See logs
    
```bash
docker logs noonewaytolearn-backend-web-1 --follow
```

Restart

```bash
docker restart noonewaytolearn-backend-web-1 
```

### Create model endpoint
GET http://localhost:5050/create_model

### Predict endpoint
GET http://localhost:5050/predict?age=31&cursus=tech&side_project=sometimes&open_source=sometimes