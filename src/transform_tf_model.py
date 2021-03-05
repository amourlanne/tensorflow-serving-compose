from tensorflow.keras.models import load_model
from tensorflow import saved_model

model = load_model("./mnist_cnn.h5")

# Check its architecture
model.summary()

tf.saved_model.save(model, "./saved_model_mnist/00000123")