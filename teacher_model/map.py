import numpy as np
import pdb

def calc_map(qB, rB, query_L, retrieval_L):
	num_query = query_L.shape[0]
	code_length = qB.shape[1]
	map = 0
	for iter in range(num_query):
		gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
		tsum = np.sum(gnd)
		if tsum == 0:
			continue
		#hamm = calc_hammingDist(qB[iter, :], rB)
		#hamm = code_length - np.sum(qB[iter]^rB, axis=1)
		hamm = np.sum(qB[iter] ^ rB, axis=1)
		# print(hamm[0:9], len(hamm))
		# pdb.set_trace()
		ind = np.argsort(hamm)
		gnd = gnd[ind]
		count = np.linspace(1, tsum, tsum)

		tindex = np.asarray(np.where(gnd == 1)) + 1.0
		map = map + np.mean(count / (tindex))
	map = map / num_query
	return map


def compute_hashing(sess, model, data, flag):
	num_data = len(data)
	batch_size = 256
	index = np.linspace(0, num_data - 1, num_data).astype(np.int32)
	# print(index)
	# pdb.set_trace()

	hash_data = []
	for i in range(num_data // batch_size + 1):
		ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]

		data_batch = data[ind]
		if flag == 'image':
			output_hash = sess.run(model.image_hash, feed_dict={model.image_data: data_batch})
		elif flag == 'text':
			output_hash = sess.run(model.text_hash, feed_dict={model.text_data: data_batch})
		hash_data.append(output_hash)
	hash_data = np.concatenate(hash_data)
	#print(hash_data.shape)
	return hash_data


def MAP(sess, model):
	train_img = np.load('/home/huhengtong/UKD/data/imgs_train.npy')
	test_img = np.load('/home/huhengtong/UKD/data/imgs_q.npy')

	train_txt = np.load('/home/huhengtong/UKD/data/texts_train.npy')
	test_txt = np.load('/home/huhengtong/UKD/data/texts_q.npy')

	train_label = np.load('/home/huhengtong/UKD/data/labels_train.npy')
	test_label = np.load('/home/huhengtong/UKD/data/labels_q.npy')

	train_img_hash = compute_hashing(sess, model, train_img, 'image')
	test_img_hash = compute_hashing(sess, model, test_img, 'image')
	train_txt_hash = compute_hashing(sess, model, train_txt, 'text')
	test_txt_hash = compute_hashing(sess, model, test_txt, 'text')

	map_i2t = calc_map(test_img_hash, train_txt_hash, test_label, train_label)
	map_t2i = calc_map(test_txt_hash, train_img_hash, test_label, train_label)
	map_i2i = calc_map(test_img_hash, train_img_hash, test_label, train_label)
	map_t2t = calc_map(test_txt_hash, train_txt_hash, test_label, train_label)

	return map_i2t, map_t2i, map_i2i, map_t2t
