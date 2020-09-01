import numpy as np
import scipy.io
import argparse
import pdb
from read_data import read_mirflickr

parser = argparse.ArgumentParser(description='Student Model Training')
parser.add_argument('--gpu', default='0', type=str, help='assign which gpu to use')
parser.add_argument('--bit', default=16, type=int, help='the bit number assigned')
args = parser.parse_args()

# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = args.gpu
per_process_gpu_memory_fraction = 0.9

# Initialize data loader
MODEL_DIR = '....../vgg19.npy'

phase = 'train'
checkpoint_dir = './checkpoint'

#SEMANTIC_EMBED = 512
#MAX_ITER = 100
num_train = 18015
batch_size = 256
image_size = 224


#images, tags, labels = loading_data(DATA_DIR)
#texts_data, labels_data, image_names = read_mirflickr()
retrieval_txt_dict = np.load(list_dir + 'train_txts_dict.npy')
test_txt_dict = np.load(list_dir + 'test_txts_dict.npy')

retrieval_label_dict = np.load(list_dir + 'train_labels_dict.npy')
test_label_dict = np.load(list_dir + 'test_labels_dict.npy')

retrieval_img_path_list = np.load(list_dir + 'train_img_path_list.npy')
test_img_path_list = np.load(list_dir + 'test_img_path_list.npy')

retrieval_x = retrieval_img_path_list
query_x = test_img_path_list 
retrieval_y = get_dict_values(retrieval_txt_dict, retrieval_img_path_list)
query_y = get_dict_values(test_txt_dict, test_img_path_list)

train_x = retrieval_x[0:15000]
train_y = retrieval_y[0:15000]

retrieval_label = get_dict_values(retrieval_label_dict, retrieval_img_path_list)
test_label = get_dict_values(test_label_dict, test_img_path_list)

train_L = retrieval_label[0:15000]

num_train = train_x.shape[0]
numClass = test_label.shape[1]
dimText = retrieval_y.shape[1]
dimLab = test_label.shape[1]


#Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999
teacher_knn_img = np.load('/....../teacher_KNN_img.npy')
teacher_knn_text = np.load('/....../teacher_KNN_text.npy')
Sim_label = (np.dot(train_label, train_label.transpose()) > 0).astype(np.int32)

Sim = np.zeros((num_train, num_train))
#ind_img = teacher_knn_img[:, 0:8000].astype(np.int32)
#ind_txt = teacher_knn_txt.astype(np.int32)
#print(ind_img.shape)
# pdb.set_trace()
#ap = 0
for i in range(num_train):
    ind = np.concatenate((teacher_knn_img[i], teacher_knn_txt[i])).astype(np.int32)
    Sim[i][ind] = 0.999


Epoch = 30

save_freq = 1

bit = args.bit
alpha = 1
gamma = 1
beta = 1
eta = 1
delta = 1

