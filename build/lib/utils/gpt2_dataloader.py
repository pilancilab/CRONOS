import numpy as np
import jax.numpy as jnp
from os.path import dirname, join, abspath

def load_data(model_name, num_batches, data_seed):
    
    # Sets seed for train-test split
    np.random.seed(data_seed)
    
    if model_name == 'gpt2_no_train':

        dataset_rel_path = join('datasets', 'gpt2notrain')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'Poslast_hidden_states_gpt2_' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_gpt2_' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0],shape_A[1]*shape_A[2])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y
        
    elif model_name == 'gpt2_notrain_meanpooled':
        
        dataset_rel_path = join('datasets', 'gpt2notrain_meanpooled')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'POSlast_hidden_states_' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0],shape_A[1])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y
    
    elif model_name == 'gpt2_notrain_commune':
        
        dataset_rel_path = join('datasets', 'gpt2notrain_commune')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'Poslast_hidden_states_gpt2XS' + str(i)+'.npy'
            training_file_name_neg = 'Neglast_hidden_states_gpt2XS' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0],shape_A[1]*shape_A[2])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y
    
    elif model_name == 'gpt2_notrain_commune_meanpooled':
        
        dataset_rel_path = join('datasets', 'gpt2notrain_commune_meanpooled')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'POSlast_hidden_states_' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0],shape_A[1])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y
    
    
    elif model_name == 'gpt2_notrain_medium':
        dataset_rel_path = join('datasets', 'gpt2notrain_medium')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'Poslast_hidden_states_gpt2_large' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_gpt2_large' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0],shape_A[1]*shape_A[2])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y

    elif model_name == 'gpt2_notrain_medium_meanpooled':
        dataset_rel_path = join('datasets', 'gpt2notrain_medium_meanpooled')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'POSlast_hidden_states_' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0], shape_A[1])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y 
    
    elif model_name == 'gpt2_lmhead_commune':
        dataset_rel_path = join('datasets', 'gpt2lmhead_commu')
        dirname(abspath('content'))
        project_root = dirname(abspath('content'))
        path = join(project_root, dataset_rel_path)
        print('path to datasets is = ', path)

        dats_training = []
        labels_training = []
        print('Loading data')
        for i in range(1, num_batches):
            training_file_name_pos = 'POSlast_hidden_states_gpt2commu_lmhead_' + str(i)+'.npy'
            training_file_name_neg = 'NEGlast_hidden_states_gpt2commu_lmhead_' + str(i)+'.npy'
            with open(join(path, training_file_name_pos), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [np.ones(X.shape[0])]
            with open(join(path, training_file_name_neg), 'rb') as ftrain:
                X = np.load(ftrain)
                dats_training += [X]
                labels_training += [-1*np.ones(X.shape[0])]
        A = jnp.concatenate([jnp.array(dat_training)
                    for dat_training in dats_training], axis=0)
        y = jnp.concatenate([jnp.array(labels)
                    for labels in labels_training], axis=0)
        print('Finished loading data!') 

        # Reshape, shuffle, and split data
        shape_A = A.shape
        A = A.reshape(shape_A[0], shape_A[1])

        n = shape_A[0]
        J = np.random.permutation(n)
        A = A[J]
        y = y[J]

        ntr = np.int64(0.8*n)
        ntst = n-ntr
        Atr = A[:ntr]
        Atst = A[ntr+1:]
        ytr = y[:ntr]
        ytst = y[ntr+1:]

        del A, y 
    
    return Atr, ytr, Atst, ytst, ntr, ntst
