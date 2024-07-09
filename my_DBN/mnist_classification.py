import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from dbn.tensorflow import SupervisedDBNClassification
from dbn.models import SupervisedDBNClassification

# Loading dataset
"""
    dictionary:
        imgs: Original images with uint8, 255
        labels: Original labels with int, 0-10.
"""
train_images=np.load('./Train_images_and_labels_array.npz')
train_img,train_labels=train_images['imgs'],train_images['labels']
test_images=np.load('./Test_images_and_labels_array.npz')
test_img,test_labels=test_images['imgs'],test_images['labels']

# Data scaling
X_train,X_test = (train_img.astype(np.float32) / 255),(test_img.astype(np.float32) / 255)
Y_train,Y_test = train_labels,test_labels
train_size,test_size = X_train.shape[0],X_test.shape[0]
X_train,X_test=np.reshape(X_train,(train_size,-1)),np.reshape(X_test,(test_size,-1))

# Data change to 
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[512,512],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='sigmoid',
                                         contrastive_divergence_iter=5,
                                         dropout_p=0.2)
classifier.fit(X_train[:500], Y_train[:500])

# Test
Y_pred = classifier.predict(X_test[:100])
print('Done.\nAccuracy: %f' % accuracy_score(Y_test[:10], Y_pred))
