import numpy as np
import sys

def sample_density(filename, shape=128):
    res_val = np.zeros((shape, shape))
    res_nnz = np.zeros((shape, shape))
    feature = np.zeros(14)

    file = open(filename)
    lines = file.readlines()
    file.close()
    if len(lines) == 0:
        return res_val, res_nnz, feature

    m, n, nnz = lines[0].split()
    m, n, nnz = int(m), int(n), int(nnz)
    if not m or not n or not nnz:
        return res_val, res_nnz, feature

    mat_val = np.zeros((m, n))
    mat_nnz = np.zeros((m, n))
    for line in lines[1:]:
        i, j, v = line.split()
        i, j, v = int(i), int(j), float(v)
        mat_val[i - 1, j - 1] = v
        mat_nnz[i - 1, j - 1] = 1

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

    if m * n <= shape * shape:
        for i in range(m):
            for j in range(n):
                cnt = i * n + j
                res_val[int(cnt / shape), cnt % shape] = mat_val[i, j]
                res_nnz[int(cnt / shape), cnt % shape] = mat_nnz[i, j]
        return res_val, res_nnz, feature

    pad = shape * shape - ((m * n) % (shape * shape))
    res_val = np.append(mat_val.reshape(-1), np.zeros(pad)).reshape(-1, shape, shape).sum(0)
    temp_nnz = np.append(mat_nnz.reshape(-1), np.zeros(pad)).reshape(-1, shape, shape)
    res_nnz = temp_nnz.sum(0) / temp_nnz.shape[0]
    return res_val, res_nnz, feature

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
    for data in data_lines:
        data_split = data.split()
        timer_list = np.zeros(4)
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
    INLIST = sys.argv[1]
    OUTLIST = sys.argv[2]
    filelist = open(INLIST).readlines()
    num = len(filelist)
    shape = 128

    images_val = np.zeros((num, shape, shape))
    images_nnz = np.zeros((num, shape, shape))
    features = np.zeros((num, 14))

    labels = read_labels(INLIST, 'data/operator/spmv_out.txt')

    # for i in range(num):
    #     filename = filelist[i].split()[0]
    #     print('[{}] {}'.format(i, filename))
    #     images_val[i], images_nnz[i], features[i] = sample_density(filename)
    # np.savez(OUTLIST, images=images_nnz, features=features, labels=labels)

    data = np.load(OUTLIST)
    print('labels', labels.shape)
    np.savez(OUTLIST, images=data['images'], features=data['features'], labels=labels)
