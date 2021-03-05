# Tensorflow Serving
```bash
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
```
# Tensorflow Serving multiple model
```bash
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8502/v1/models/half_plus_two:predict
    
curl -d '{"instances": img_shape(1,28,28,1) }' \
    -X POST http://localhost:8502/v1/models/mnist:predict
```
# Serving with Flask app
http://localhost:5000/