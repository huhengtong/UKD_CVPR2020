import linecache, pdb
import numpy as np


def push_query(query, url, dict):
	if query in dict:
		dict[query].append(url)
	else:
		dict[query] = [url]
	return dict


def make_train_dict(query_list, url_list, KNN_cross):
	query_url = {}
	query_pos = {}
	query_neg = {}
	query_num = len(query_list)
	#url_num = len(url_list)
	
	for i in range(query_num):
		query = query_list[i]
		pos_idx = KNN_cross[i].astype(np.int32)
		#print(query)
		# print(url_list[pos_idx])
		query_pos[query] = np.array(url_list)[pos_idx]
		query_neg[query] = np.delete(url_list, pos_idx)
		query_url[query] = url_list

	return query_url, query_pos, query_neg


def make_test_dict(query_list, url_list, query_label, url_label):
	query_url = {}
	query_pos = {}
	query_num = len(query_list)
	url_num = len(url_list)
	
	for i in range(query_num):
		query = query_list[i]
		for j in range(url_num):
			url = url_list[j]				
			if np.dot(query_label[i], url_label[j]) > 0:
				push_query(query, url, query_url)
				push_query(query, url, query_pos)
			else:
				push_query(query, url, query_url)
	return query_url, query_pos		


def load_all_query_url():
	#train_img = np.load('/home/huhengtong/UKD/imgs_train.npy')
	#test_img = np.load('/home/huhengtong/UKD/imgs_q.npy')

	#train_txt = np.load('/home/huhengtong/UKD/texts_train.npy')
	#test_txt = np.load('/home/huhengtong/UKD/texts_q.npy')

	train_img_idx = range(18015)
	train_txt_idx = range(20015, 18015+20015)

	KNN_cross = np.load('/home/huhengtong/UKD/data/' + 'KNN_cross5.npy')
	
	train_i2t, train_i2t_pos, train_i2t_neg = make_train_dict(train_img_idx, train_txt_idx, KNN_cross)
	train_t2i, train_t2i_pos, train_t2i_neg = make_train_dict(train_txt_idx, train_img_idx, KNN_cross)

	return train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg


def load_all_feature():
	#feature_dict = {}
	train_img = np.load('/home/huhengtong/UKD/data/imgs_train.npy')
	test_img = np.load('/home/huhengtong/UKD/data/imgs_q.npy')

	train_txt = np.load('/home/huhengtong/UKD/data/texts_train.npy')
	test_txt = np.load('/home/huhengtong/UKD/data/texts_q.npy')

	img_data = np.concatenate((train_img, test_img))
	txt_data = np.concatenate((train_txt, test_txt))
	idx_img = range(20015)
	idx_txt = range(20015, 40030)
	dict_img = dict(zip(idx_img, img_data))
	dict_txt = dict(zip(idx_txt, txt_data))
	dict_txt.update(dict_img)

	return dict_txt


def load_all_label():
	train_label = np.load('/home/huhengtong/UKD/data/labels_train.npy')
	test_label = np.load('/home/huhengtong/UKD/data/labels_q.npy')

	label_data = np.concatenate((train_label, test_label))
	idx_label_img = range(20015)
	idx_label_txt = range(20015, 40030)

	label_img_dict = dict(zip(idx_label_img, label_data))
	label_txt_dict = dict(zip(idx_label_txt, label_data))
	label_txt_dict.update(label_img_dict)

	return label_txt_dict
	
	
