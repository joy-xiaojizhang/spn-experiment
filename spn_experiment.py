from tachyon.SPN2 import SPN
import numpy as np

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('output/movietrain.txt', 'train', cont=True)
spn.add_data('output/movietest.txt', 'test', cont=True)
# create a valid sum product network

sum_branch_factor = (2, 4)
prod_branch_factor = (20, 40)
num_variables = 1000

spn.make_random_model((prod_branch_factor, sum_branch_factor), num_variables, cont=True)

# start the session
spn.start_session()

# train
epochs = 1
# access the data
train = spn.data.train[:, :1000]
test = spn.data.test
spn.train(epochs, train, minibatch_size=100)
test_loss = spn.evaluate(test, cond_probs=np.zeros(test.shape[:2]))
print('Loss:', test_loss)
# Loss: 6.263
