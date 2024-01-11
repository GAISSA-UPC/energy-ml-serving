import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import random

fashion_mnist=keras.datasets.fashion_mnist
(_, _), (x_test, y_test) = fashion_mnist.load_data()
assert len(x_test) == 10000
assert len(y_test) == 10000


dataset = "fashion"
label_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

saved_model_dir = f"models/model_{dataset}.h5"

# load model
try:
    print(saved_model_dir)
    model = keras.models.load_model(saved_model_dir)
    print("Model loaded correctly")
except:
    print("There is a problem with the file path")
    
def see_x_image(x,y,name=None,caption=True, save_dir="."):
    '''
    See image
    '''
    plt.figure()
    
    plt.imshow((x.reshape((28,28))).astype("uint8"))
    title=str(y)
    if name:
        title += " "+name
        plt.title(title)
    if caption:
        plt.title(title)
    print(save_dir)
    plt.savefig(save_dir+"/"+dataset+"_image"+ ".png")
    plt.axis("off")
    
    
# Get random number between 0 and len(x_test)
ran = random.randint(0, len(x_test))
print(y_test[ran])
#print(list(y_test[ran]).index(max(y_test[ran])))
label_name = label_names[y_test[ran]]
see_x_image(x_test[ran],y_test[ran],label_name,save_dir="D:/GAISSA/deploy-GAISSA/scripts/")

# Inference
# predict with that random
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_test[ran:ran+1].shape)
pred = tf.keras.utils.to_categorical(np.argmax(model.predict(x_test[ran:ran+1])), 10)
cat_pred = list(pred).index(max(pred))
print("Prediction: ",pred, ", ",cat_pred)
print("Prediction clothes: ", label_names[cat_pred])


wrong = []
for i in range(20):
    # Get random number between 0 and len(x_test)
    ran = random.randint(0, len(x_test))
    print(y_test[ran])
    #print(list(y_test[ran]).index(max(y_test[ran])))
    label_name = label_names[y_test[ran]]
    #see_x_image(x_test[ran],y_test[ran],label_name,save_dir="D:/GAISSA/deploy-GAISSA/scripts/")

    # Inference
    # predict with that random
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_test[ran:ran+1].shape)
    pred = tf.keras.utils.to_categorical(np.argmax(model.predict(x_test[ran:ran+1])), 10)
    cat_pred = list(pred).index(max(pred))
    print("Prediction: ",pred, ", ",cat_pred)
    print("Prediction clothes: ", label_names[cat_pred])
    if (y_test[ran] != cat_pred):
        wrong.append(i)

print(wrong)