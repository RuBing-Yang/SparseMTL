import requests
import re
import wget
import os
import glob
import tarfile
import numpy as np
import random

def get_download_urls():
    download_urls = []
    file = open('download_urls.txt', 'a')
    for i in range(1, 146):
        url = 'http://sparse.tamu.edu/' + '/?page=' + str(i)
        html_text = requests.get(url).text
        # for html_line in html_text.splitlines():
        page_download_urls = re.findall("https://suitesparse-collection-website.herokuapp\.com/MM/[0-9a-zA-Z_/-]+\.tar\.gz", html_text)
        file.writelines('\n'.join(page_download_urls) + '\n')
        download_urls.extend(page_download_urls)
        print("page", i, "list num", len(page_download_urls), page_download_urls)
    print("urls num", len(download_urls))
    file.close()

def download_files():
    file = open("download_urls.txt")
    directory = "suite-sparse/origin/"
    download_urls = file.readlines()
    file.close()
    count = 0
    for url in download_urls:
        count += 1
        file_name = url.split('/')[-1]
        if os.path.isfile(directory + file_name):
            print("[{}] {} already exists".format(count, file_name))
            continue
        print("[{}] {}".format(count, file_name))
        # response = requests.get(url)
        # file = open("suite-sparse/origin/" + file_name, "wb")
        # file.write(response.content)
        # file.close()
        for attempt in range(5):
            try:
                wget.download(url, directory + file_name)
            except:
                print("{} attempt error, retry {}".format(file_name, attempt))
            else:
                break

def unzip_files():
    for path, directories, files in os.walk('suite-sparse/origin'):
        count = 0
        print("tar files num:", len(files))
        for zip_file_name in files:
            count += 1
            zip_file_path = os.path.join(path, zip_file_name)
            zip_file_size = os.path.getsize(zip_file_path)
            if count % 100 == 0:
                print("[{}] zip file path: {}".format(count, zip_file_path))
            if zip_file_size > 1024 * 1024 * 20:
                continue  # 跳过大文件
            try:
                with tarfile.open(zip_file_path, 'r:gz') as tar:
                    for member in tar:
                        if member.isdir():
                            continue
                        mtx_file_name = member.name.rsplit('/',1)[1]
                        if os.path.isfile('suite-sparse/unzip/' + mtx_file_name):
                            continue
                        print("[{}] unzip file: {} size {}".format(count, mtx_file_name, zip_file_size))
                        tar.makefile(member, 'suite-sparse/unzip/' + mtx_file_name)
            except Exception as e:
                print("[{}] unzip file {} size {} error: {}".format(count, mtx_file_name, zip_file_size, str(e)))

def get_file_list(folder='/home/rubing/SpTFS/data/suite-sparse/unzip/', output='suite-sparse/small_file_list.txt'):
    output_file = open(output, 'w')
    for path, directories, files in os.walk(folder):
        count = 0
        print("mtx files num:", len(files))
        for mtx_file_name in files:
            count += 1
            if mtx_file_name.split('.')[-1] != 'mtx':
                continue
            mtx_file_path = folder + mtx_file_name
            mtx_file_size = os.path.getsize(mtx_file_path)
            if count % 100 == 0:
                print("[{}] mtx file path: {}".format(count, mtx_file_path))
            # if mtx_file_size > 1024 * 1024 * 10:
            #     continue
            if mtx_file_size > 1024 * 1024 * 1:
                continue
            output_file.write(mtx_file_path + '\n')
    output_file.close()

def delete_mtx_head():
    input_folder = '/home/rubing/SpTFS/data/suite-sparse/unzip/'
    output_folder = '/home/rubing/SpTFS/data/suite-sparse/nohead/'
    for path, directories, files in os.walk(input_folder):
        count = 0
        print("mtx files num:", len(files))
        for file_name in files:
            if file_name.split('.')[-1] != 'mtx':
                continue
            if os.path.isfile(output_folder + file_name):
                continue
            if os.path.getsize(input_folder + file_name) > 1024 * 1024 * 10:
                continue  # 跳过大文件
            input_file = open(input_folder + file_name, 'r')
            lines = input_file.readlines()
            input_file.close()
            # TODO: support symmetry=symmetric(DIA) and format=array
            # head, dimesion, formats, field, symmetry = lines[0].split(' ')
            # if head != '%%MatrixMarket' or dimesion != 'matrix' or field != 'real' or \
            #     formats != 'coordinate' or symmetry != 'general': 
            symmetric = 'symmetric' in lines[0]
            if 'MatrixMarket matrix coordinate real' in lines[0]:
                head_cnt = 0
                for line in lines:
                    if line[0] != '%':
                        break
                    head_cnt += 1
                lines = lines[head_cnt:]
            elif 'MatrixMarket matrix array real' in lines[0]:
                new_lines = []
                index = 0
                m, n, nnz = 1, 1, 0
                for line in lines:
                    if line[0] == '%':
                        continue
                    if index == 0:
                        m, n = line.split()
                        m, n = int(m), int(n)
                        new_lines.append('{} {}'.format(m, n))
                        # print('m, n:', m, n)
                    else:
                        v = line.split()[0]
                        if v != '1e308' and v != '0':
                            nnz += 1
                            # print('[{}] n={}: {} {} {}'.format(nnz, n, int(index / n), int((index - 1) % n), v))
                            new_lines.append('{} {} {}\n'.format(int((index - 1) / n) + 1, int((index - 1) % n) + 1, v))
                    index += 1
                # if nnz == 0:
                #     continue
                new_lines[0] = '{} {}\n'.format(new_lines[0], nnz)
                # print('first line:', new_lines[0])
                lines = new_lines
            else:
                # print("{}{}".format(lines[0], input_folder + file_name))
                continue
            if symmetric:
                m, n, nnz = lines[0].split()
                m, n, nnz = int(m), int(n), int(nnz)
                tmp_lines = lines[1:]
                for line in tmp_lines:
                    i, j, v = line.split()
                    if i != j:
                        lines.append('{} {} {}\n'.format(j, i, v))
                        nnz += 1
                lines[0] = '{} {} {}\n'.format(m, n, nnz)
            output_file = open(output_folder + file_name, 'w')
            output_file.writelines(lines)
            output_file.close()
            count += 1
            print("[{}] mtx file path: {}".format(count, input_folder + file_name))
    
def label_batch(tensorlist_filename, scorelist_filename, outputlist_filename):
    tensorlist_file = open(tensorlist_filename, 'r')
    tensorlist = [i.split()[0] for i in tensorlist_file.readlines()]
    print('tensorlist', len(tensorlist))
    tensorlist_file.close()
    scorelist_file = open(scorelist_filename, 'r')
    scorelist = scorelist_file.readlines()
    print('scorelist', len(scorelist))
    scorelist_file.close()
    outputlist = []
    for scoreline in scorelist:
        filename = scoreline.split()[0].replace('unzip', 'nohead').replace('crop_head', 'crop')
        if filename in tensorlist:
            outputlist.append(filename)
        else:
            print('score', filename, 'not exit in filename')
    print('outputlist', len(outputlist))
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def find_all_zero(folder='/home/rubing/SpTFS/data/suite-sparse/nohead/', output_filename='suite-sparse/all_zero_list.txt'):
    output_filelist = []
    count = 0
    for path, directories, files in os.walk(folder):
        print("mtx files num:", len(files))
        for file_name in files:
            file_path = folder + file_name
            print("[{}] {}".format(count, file_path))
            count += 1
            mtx_file = open(file_path, 'r')
            line = mtx_file.readline()
            if line == "" or len(line) == 0:
                output_filelist.append(file_path)
                continue
            m, n, nnz = line.split()
            if nnz == "0":
                output_filelist.append(file_path)
    output_file = open(output_filename, 'w')
    output_file.write('\n'.join(output_filelist))
    output_file.close()

def read_labels(filelist_filename='suite-sparse/all_zero_list.txt', date_filename='/home/rubing/taco/tools/output/spmv_out.txt'):
    filelist_file = open(filelist_filename, 'r')
    file_list = [i.replace('\n', '') for i in filelist_file.readlines()]
    filelist_file.close()
    data_file = open(date_filename, 'r')
    data_lines = data_file.readlines()
    data_file.close()

    label_map = {}
    labels = []
    format_list = ['DM', 'COO', 'CSR', 'CSC', 'DCSR', 'DCSC']
    for data in data_lines:
        data_split = data.split()
        time_list = np.zeros(6)
        for i in range(6):
            for j in range(3):
                time_list[i] += float(data_split[i * 3 + j + 1])
        label_map[data_split[0]] = format_list[np.argmin(time_list)]
    for file_name in file_list:
        _file_name = file_name.replace('nohead', 'unzip')
        if _file_name in label_map.keys():
            labels.append(label_map[_file_name])
    print(labels)
    return labels

def file_devide(front_file='suite-sparse/spmv_sddmm_list.txt', behind_file='suite-sparse/all_zero_list.txt', output_file='suite-sparse/no_all_zero_list.txt'):
    front_lines = [i.split()[0] for i in open(front_file, 'r').readlines()]
    behind_lines = [i.split()[0].split('/')[-1] for i in open(behind_file, 'r').readlines()]
    print('front_lines', len(front_lines))
    print('behind_lines', len(behind_lines))
    output_lines = []
    for line in front_lines:
        if line.split('/')[-1] not in behind_lines:
            output_lines.append(line)
    print('output_lines', len(output_lines))
    open(output_file, 'w').write('\n'.join(output_lines))

def random_crop(inputlist_filename='suite-sparse/no_all_zero_list.txt', outputlist_filename='suite-sparse/crop_no_all_zero_list.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    for filename_line in inputlist:
        filename = filename_line.split()[0]
        mtx_file = open(filename)
        lines = mtx_file.readlines()
        mtx_file.close()
        m, n, nnz = lines[0].split()
        m, n, nnz = int(m), int(n), int(nnz)
        if m * n < 20 or nnz < 10: #设置threshold
            continue
        num = 1
        if nnz > 10000:
            num = 6
        elif nnz > 5000:
            num = 4
        elif nnz > 3000:
            num = 3
        elif nnz > 1000:
            num = 2
        for count in range(num):
            output_lines = []
            new_m = np.random.randint(1, m + 1)
            new_n = np.random.randint(1, n + 1)
            start_m = np.random.randint(1, m - new_m + 2)
            start_n = np.random.randint(1, n - new_n + 2)
            end_m = start_m + new_m - 1
            end_n = start_n + new_n - 1
            for line in lines[1:]:
                i, j, v = line.split()
                i, j, v = int(i), int(j), float(v)
                if start_m <= i and i <= end_m and start_n <= j and j <= end_n:
                    output_lines.append('{} {} {}'.format(i - start_m + 1, j - start_n + 1, v))
            if len(output_lines) > 0:
                output_lines.insert(0, '{} {} {}'.format(new_m, new_n, len(output_lines)))
                crop_filename = '{}_{}.mtx'.format(filename.replace('nohead', 'crop').split('.mtx')[0], count)
                print(crop_filename, output_lines[0])
                output_file = open(crop_filename, 'w')
                output_file.write('\n'.join(output_lines))
                output_file.close()
                outputlist.append(crop_filename)
                output_lines.insert(0, '%%MatrixMarket matrix coordinate real general')
                output_file_head = open(crop_filename.replace('crop', 'crop_head'), 'w')
                output_file_head.writelines('\n'.join(output_lines))
                output_file_head.close()

    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def active_threshold(inputlist_filename, outputlist_filename):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    for filename_line in inputlist:
        filename = filename_line.split()[0]
        mtx_file = open(filename.replace('unzip', 'nohead').replace('crop_head', 'crop'))
        m, n, nnz = mtx_file.readline().split()
        mtx_file.close()
        nnz = int(nnz)
        if nnz > 200 and nnz < 10000:
            outputlist.append(filename)
    print('outputlist', len(outputlist))
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def cal_nnz(inputlist_filename='data/generate-dataset/dia_list.txt', outputlist_filename='data/generate-dataset/dia_list_new.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    all_list = []
    outputlist = []
    for i in range(len(inputlist)):
        filename = inputlist[i].split()[0]
        if not os.path.isfile(filename):
            continue
        print(filename)
        mtx_file = open(filename, 'r')
        m, n, nnz = mtx_file.readline().split()
        m, n, nnz = int(m), int(n), int(nnz)
        all_list.append([m, n ,nnz])
        if nnz > 0:
            outputlist.append(filename)
    all_list = np.matrix(all_list)
    print('all_list', all_list.shape)
    print('avg', np.mean(all_list[:,0], axis=0), np.mean(all_list[:,1], axis=0), np.mean(all_list[:,2], axis=0))
    print('mid', np.median(all_list[:,0], axis=0), np.median(all_list[:,1], axis=0), np.median(all_list[:,2], axis=0))

    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def get_coo(inputlist_filename='data/suite-sparse/no_all_zero_list.txt', outputlist_filename='data/generate-dataset/coo_list.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    count = 0
    for i in range(len(inputlist)):
        filename = inputlist[i].split()[0]
        mtx_file = open(filename, 'r')
        m, n, nnz = mtx_file.readline().split()
        m, n, nnz = int(m), int(n), int(nnz)
        if m == n:
            continue
        if nnz < 150 or nnz > 10000:
            continue
        if nnz > m * n / 1.5:
            continue
        print('[{}] {}'.format(count, filename))
        outputlist.append(filename)
        count += 1
        if count > 1500:
            break
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def get_dense(inputlist_filename='data/suite-sparse/no_all_zero_list.txt', outputlist_filename='data/generate-dataset/dense_list.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    count = 0
    for i in range(len(inputlist)):
        filename = inputlist[i].split()[0]
        mtx_file = open(filename, 'r')
        m, n, nnz = mtx_file.readline().split()
        m, n, nnz = int(m), int(n), int(nnz)
        if m == n:
            continue
        if nnz < 150 or nnz > 10000:
            continue
        if nnz < m * n  * 0.95:
            continue
        print('[{}] {}'.format(count, filename))
        outputlist.append(filename)
        count += 1
        if count > 1500:
            break
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def get_symmetric(outputlist_filename='data/generate-dataset/dia_origin_list.txt'):
    input_folder = '/home/rubing/SpTFS/data/suite-sparse/unzip/'
    outputlist = []
    for path, directories, files in os.walk(input_folder):
        count = 0
        print("mtx files num:", len(files))
        for file_name in files:
            if file_name.split('.')[-1] != 'mtx':
                continue
            file_path = input_folder + file_name
            input_file = open(file_path, 'r')
            line = input_file.readline()
            input_file.close()
            if not'symmetric' in line:
                continue
            nohead_file_path = file_path.replace('unzip', 'nohead')
            if not os.path.isfile(nohead_file_path):
                continue
            mtx_file = open(nohead_file_path, 'r')
            m, n, nnz = mtx_file.readline().split()
            print(file_path, m, n, nnz)
            m, n, nnz = int(m), int(n), int(nnz)
            if m == n:
                continue
            if nnz < 100 or nnz > 5000:
                continue
            if nnz > m * n / 1.5:
                continue
            outputlist.append(file_path)
    print('outputlist', len(outputlist))
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close() 

def gen_symmetric(inputlist_filename='data/suite-sparse/no_all_zero_list.txt', outputlist_filename='data/generate-dataset/dia_gen_list.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    count = 0
    for i in range(len(inputlist)):
        filename = inputlist[i].split()[0]
        mtx_file = open(filename, 'r')
        m, n, nnz = mtx_file.readline().split()
        m, n, nnz = int(m), int(n), int(nnz)
        if m == n:
            continue
        if nnz < 100 or nnz > 5000:
            continue
        if nnz > m * n / 1.4:
            continue
        print('[{}] {}'.format(count, filename))
        lines = mtx_file.readlines()
        value_map = {}
        for line in lines:
            i, j, v = line.split()
            i, j, v = int(i), int(j), float(v)
            value_map['{}_{}'.format(i, j)] = v
        for line in lines:
            i, j, v = line.split()
            i, j, v = int(i), int(j), float(v)
            if '{}_{}'.format(j, i) not in value_map.keys():
                lines.append('{} {} {}\n'.format(j, i, v))
                nnz += 1
        if nnz > m * n / 1.5:
            continue
        dim = max(m, n)
        lines.insert(0, '{} {} {}\n'.format(dim, dim, nnz))
        new_filename = filename.replace('.mtx', '_symmetric.mtx')
        new_file = open(new_filename, 'w')
        new_file.writelines(lines)
        new_file.close()
        outputlist.append(new_filename)
        count += 1
        if count > 1100:
            break
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def gen_dense(outputlist_filename='data/generate-dataset/dense_gen_list.txt'):
    count = 0
    outputlist = []
    while count < 800:
        m = np.random.randint(1, 9500)
        n = np.random.randint(1, 9500)
        length = m * n
        if length > 12000 or length < 150:
            continue
        values = np.random.random(length) * 100 - 50
        zero_ids = random.sample(range(length), np.random.randint(length * 0.001, length * 0.03))
        nnz = length - len(zero_ids)
        lines = ['{} {} {}'.format(m, n, nnz)]
        for i in range(length):
            if i not in zero_ids:
                lines.append('{} {} {}'.format(int(i / n) + 1, i % n + 1, values[i]))
        filename = '/home/rubing/SparseMTL/data/generate-dataset/dense/gen_{}.mtx'.format(count)
        print(filename, m, n, nnz)
        mtx_file = open(filename, 'w')
        mtx_file.write('\n'.join(lines))
        mtx_file.close()
        outputlist.append(filename)
        count += 1
    outputlist_file = open(outputlist_filename, 'w')
    outputlist_file.write('\n'.join(outputlist))
    outputlist_file.close()

def append_head(inputlist_filename='data/generate-dataset/dia_gen_list.txt'):
    inputlist_file = open(inputlist_filename, 'r')
    inputlist = inputlist_file.readlines()
    inputlist_file.close()
    outputlist = []
    count = 0
    for i in range(len(inputlist)):
        filename = inputlist[i].split()[0]
        print(filename)
        lines = []
        with open(filename, 'r') as mtx_file:
            lines = mtx_file.readlines()
        lines.insert(0, '%%MatrixMarket matrix coordinate real general\n')
        with open(filename, 'w') as mtx_file:
            mtx_file.writelines(lines)
       

if __name__ == '__main__':
    # get_file_list()
    # delete_mtx_head()
    # find_all_zero()
    # print(read_labels())
    
    # get_file_list('/home/rubing/SpTFS/data/suite-sparse/nohead/', 'suite-sparse/nohead_list.txt')
    # label_batch('data/suite-sparse/nohead_list.txt', 'data/operator/spmv_out.txt', 'data/operator/spmv_list.txt')
    # label_batch('data/suite-sparse/spmv_list.txt', 'data/operator/sddmm_out.txt', 'data/operator/spmv_sddmm_list.txt')
    # label_batch('data/suite-sparse/nohead_with_crop_list.txt', 'data/operator/spmv_out.txt', 'data/operator/spmv_crop_list.txt')
    # label_batch('data/operator/spmv_crop_list.txt', 'data/operator/sddmm_out.txt', 'data/operator/spmv_sddmm_crop_list.txt')
    # active_threshold('data/operator/spmv_sddmm_crop_list.txt', 'data/operator/active_spmv_sddmm_crop_list.txt')
    
    # file_devide()
    # file_devide('data/operator/spmv_out.txt', 'data/operator/sddmm_out.txt', 'data/operator/spmv_devide_sddmm.txt')
    # file_devide('data/suite-sparse/nohead_list.txt', 'data/operator/spmv_out.txt', 'data/operator/nohead_devide_spmv.txt')

    # random_crop()
    # crop_add_head()
    # get_file_list('/home/rubing/SparseMTL/data/suite-sparse/crop_head/', 'suite-sparse/crop_small_file_list.txt')

    # get_symmetric()
    # cal_nnz()
    # gen_symmetric()
    # get_coo()
    # get_dense()
    # gen_dense()
    append_head()