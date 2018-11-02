import h5py


def read_h5_file(folder_path,file_name):

    filename = file_name + '.h5'
    f = h5py.File(filename, 'r')

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])