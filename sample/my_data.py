import numpy as np
import sys

def sample_histogram(filename, shape=128):
    image_dim0 = np.zeros((shape, shape))
    image_dim1 = np.zeros((shape, shape))
    feature = np.zeros(14)

    file = open(filename)
    lines = file.readlines()
    file.close()
    if len(lines) == 0:
        return image_dim0, image_dim1, feature

    m, n, nnz = lines[0].split()
    m, n, nnz = int(m), int(n), int(nnz)
    print(filename, m, n, nnz)
    if not m or not n or not nnz:
        return image_dim0, image_dim1, feature

    ratio_dim0 = shape / m
    ratio_dim1 = shape / n
    max_dim = max(m, n)
    mat_val = np.zeros((m, n))
    mat_nnz = np.zeros((m, n))
    # print('ratio_dim0', ratio_dim0, 'm', m)
    # print('ratio_dim0', ratio_dim0, 'n', n)

    for line in lines[1:]:
        i, j, v = line.split()
        i, j, v = int(i), int(j), float(v)
        mat_val[i - 1, j - 1] = v
        mat_nnz[i - 1, j - 1] = 1
        dist = int(shape * abs(i - j) / max_dim)
        map_dim0 = min(int(i * ratio_dim0), 127)
        # print('i', i, 'ratio_dim0', ratio_dim0, 'map_dim0', map_dim0)
        map_dim1 = min(int(j * ratio_dim1), 127)
        # print('j', j, 'ratio_dim1', ratio_dim1, 'map_dim1', map_dim1)
        image_dim0[map_dim0][dist] += 1
        image_dim1[map_dim1][dist] += 1

    nnz_dim0, nnz_dim1 = mat_nnz.sum(0), mat_nnz.sum(1)
    dens = nnz / (m * n)
    feature = np.array([
        m, n, nnz, dens,
        nnz_dim0.mean(), nnz_dim1.mean(),
        nnz_dim0.max(), nnz_dim1.max(),
        nnz_dim0.min(), nnz_dim1.min(),
        nnz_dim0.max() - nnz_dim0.min(), nnz_dim1.max() - nnz_dim1.min(),
        nnz_dim0.std(), nnz_dim1.std()
    ])
    return image_dim0, image_dim1, feature

def read_labels(filelist_filename, data_filename):
    filelist_file = open(filelist_filename, 'r')
    file_list = [i.replace('\n', '') for i in filelist_file.readlines()]
    filelist_file.close()
    print(filelist_filename, len(file_list))
    data_file = open(data_filename, 'r')
    data_lines = data_file.readlines()
    data_file.close()
    print(data_filename, len(data_lines))
    
    label_map = {}
    labels = []
    label_name = ['DM', 'COO', 'CSR', 'DCSR']
    # DM, COO, CSR, CSC, DCSR, DCSC
    # DM, COO, CSR
    active_labels = {0: 0, 1: 1, 2: 2, 4: 3}
    format_num = len(active_labels.keys())
    for data in data_lines:
        data_split = data.split()
        timer_list = np.zeros(format_num)
        for i in range(6):
            timer = 0
            for j in range(3):
                timer += float(data_split[i * 3 + j + 1])
            # timer_list[i] = timer
            if i in active_labels.keys():
                timer_list[active_labels[i]] = timer
        nohead_filename = data_split[0].replace('unzip', 'nohead').replace('crop_head', 'crop')
        label_map[nohead_filename] = np.argmin(timer_list)
    count = 0
    for file_name in file_list:
        file_name = file_name.split()[0]
        label = label_map[file_name]
        labels.append(label)
        # print('[{}] {}: {}'.format(count, file_name, label_name[label]))
        count += 1
    # print(labels)
    return np.array(labels)


if __name__ == '__main__':
    OUTLIST = sys.argv[1]

    dia_list = open('data/generate-dataset/dia_gen_list.txt', 'r').readlines()
    dense_list = open('data/generate-dataset/dense_list.txt', 'r').readlines()
    coo_list = open('data/generate-dataset/coo_list.txt', 'r').readlines()

    filelist = dia_list + dense_list + coo_list
    labels = np.concatenate((np.zeros(len(dia_list)), np.ones(len(dense_list)), np.ones(len(coo_list)) * 2))
    labels = np.matrix([labels, labels])

    num = len(filelist)
    shape = 128
    print('num', num)
    print('labels', labels.shape)

    images_dim0 = np.zeros((num, shape, shape))
    images_dim1 = np.zeros((num, shape, shape))
    features = np.zeros((num, 14))

    for i in range(num):
        filename = filelist[i].split()[0]
        print('[{}] {}'.format(i, filename))
        images_dim0[i], images_dim1[i], features[i] = sample_histogram(filename)
    np.savez(OUTLIST, images_dim0=images_dim0, images_dim1=images_dim1, features=features, labels=labels)

    print('finished!!!')
