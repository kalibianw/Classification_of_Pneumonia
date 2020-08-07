from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

from utils import TrainModule

nploader = np.load("../APP/Virus_all_(400,500).npz")

x_data, y_data = np.expand_dims(nploader["x_data"], axis=-1), to_categorical(nploader["label"])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
print(
    np.shape(x_train),
    np.shape(x_test),
    np.shape(y_train),
    np.shape(y_test)
)

tm = TrainModule(ckpt_path="ckpt/Virus_all_(400,500)_train.ckpt",
                 model_save_path="model/Virus_all_(400,500)_train.h5",
                 input_shape=np.shape(x_train)[1:],
                 result_file_name="Virus_all_(400, 500)_training_result"
                 )

model = tm.create_model()
model.summary()

tm.model_training(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)
