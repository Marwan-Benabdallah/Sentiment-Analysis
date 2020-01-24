<<<<<<< HEAD
import os
import math
import time
import pickle
import argparse
import torch 
from torchtext import data
from torchtext import datasets
import random
from datetime import datetime
import torch.optim as optim
import gensim

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import time

from utils import build_w2c, build_w2i, build_dataset,build_batch, associate_parameters, forwards, sort_data_by_length, binary_pred, make_emb_zero, init_V,binary_accuracy
from layers import CNN

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)



RESULTS_DIR = './results/' + datetime.now().strftime('%Y%m%d%H%M')
try:
    os.mkdir('results')
except:
    pass
try:
    os.mkdir(RESULTS_DIR)
except:
    pass




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model,train_x,valid_x, optimizer, criterion):
    
    optimizer.zero_grad()
        
    predictions = model(train_x).squeeze(1)
    loss = criterion(predictions, valid_x)
    acc = binary_accuracy(predictions, valid_x)
    loss.backward()
        
    optimizer.step()
        
    epoch_loss += loss.item()
    epoch_acc += acc.item()
        
    return epoch_loss , epoch_acc 


# def evaluate(model, iterator, criterion):
    
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.eval()
    
#     with torch.no_grad():
    
#         for batch in iterator:

#             predictions = model(batch.text).squeeze(1)
            
#             loss = criterion(predictions, batch.label)
            
#             acc = binary_accuracy(predictions, batch.label)

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = 2
    WIN_SIZES = [3,4,5]
    BATCH_SIZE = 64
    EMB_DIM = 300
    OUT_DIM = 1
    L2_NORM_LIM = 3.0
    NUM_FIL = 100
    DROPOUT_PROB = 0.5
    V_STRATEGY = 'static'
    ALLOC_MEM = 4096

    if V_STRATEGY in ['rand', 'static', 'non-static']:
        NUM_CHA = 1
    else:
        NUM_CHA = 2

    # FILE paths
    W2V_PATH     = 'GoogleNews-vectors-negative300.bin'
    TRAIN_X_PATH = 'train_x.txt'
    TRAIN_Y_PATH = 'train_y.txt'
    VALID_X_PATH = 'valid_x.txt'
    VALID_Y_PATH = 'valid_y.txt'


    # Load pretrained embeddings
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True,limit=1)
    vocab = pretrained_model.wv.vocab.keys()
    w2v = pretrained_model.wv

    # Build dataset =======================================================================================================
    w2c = build_w2c(TRAIN_X_PATH, vocab=vocab)
    w2i, i2w = build_w2i(TRAIN_X_PATH, w2c, unk='unk')
    train_x, train_y = build_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, w2i, unk='unk')
    valid_x, valid_y = build_dataset(VALID_X_PATH, VALID_Y_PATH, w2i, unk='unk')
    train_x, train_y = sort_data_by_length(train_x, train_y)
    valid_x, valid_y = sort_data_by_length(valid_x, valid_y)
    VOCAB_SIZE = len(w2i)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    print('VOCAB_SIZE:', VOCAB_SIZE)
    
    V_init = init_V(w2v, w2i)
    
    train_iterator, valid_iterator= data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE, 
    device = device)

    with open(os.path.join(RESULTS_DIR, './w2i.dump'), 'wb') as f_w2i, open(os.path.join(RESULTS_DIR, './i2w.dump'), 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    # Build model =================================================================================
 
    model=CNN(VOCAB_SIZE, EMB_DIM, NUM_FIL, WIN_SIZES, OUT_DIM, 
                 DROPOUT_PROB, len(w2i))


    # Train model ================================================================================
   
    pretrained_embeddings = torch.tensor(V_init)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[len(w2i)] = torch.zeros(EMB_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)    
    criterion = criterion.to(device)
    n_batches_train = math.ceil(len(train_x)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_x)/BATCH_SIZE)
    
    best_valid_loss = float('inf')

    for i in range(N_EPOCHS):


        start_time = time.time()
        epoch_loss = 0
        epoch_acc = 0 
        epoch_loss = 0
        epoch_acc = 0
  
  
    
        for i in range(n_batches_train):
        
    
            start = i*BATCH_SIZE
            end = start+BATCH_SIZE      
            train_loss, train_acc = train(model,train_x,train_y, optimizer, criterion)
            #valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut4-model.pt')
        
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

if __name__ == '__main__':
    main()
=======
import os
import math
import time
import pickle
import argparse
import torch 
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import random
from datetime import datetime
import torch.optim as optim
import gensim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import time

from utils import build_w2c, build_w2i, build_dataset,build_batch, associate_parameters, forwards, sort_data_by_length, binary_pred, make_emb_zero, init_V, binary_accuracy
from layers import CNN

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)



RESULTS_DIR = './results/' + datetime.now().strftime('%Y%m%d%H%M')
try:
    os.mkdir('results')
except:
    pass
try:
    os.mkdir(RESULTS_DIR)
except:
    pass




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = 2
    WIN_SIZES = [3,4,5]
    BATCH_SIZE = 64
    EMB_DIM = 300
    OUT_DIM = 1
    L2_NORM_LIM = 3.0
    NUM_FIL = 100
    DROPOUT_PROB = 0.5
    V_STRATEGY = 'static'
    ALLOC_MEM = 4096

    if V_STRATEGY in ['rand', 'static', 'non-static']:
        NUM_CHA = 1
    else:
        NUM_CHA = 2

    # FILE paths
    W2V_PATH     = 'GoogleNews-vectors-negative300.bin'
    TRAIN_X_PATH = 'train_x.txt'
    TRAIN_Y_PATH = 'train_y.txt'
    VALID_X_PATH = 'valid_x.txt'
    VALID_Y_PATH = 'valid_y.txt'


    # Load pretrained embeddings
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
    vocab = pretrained_model.wv.vocab.keys()
    w2v = pretrained_model.wv

    # Build dataset =======================================================================================================
    w2c = build_w2c(TRAIN_X_PATH, vocab=vocab)
    w2i, i2w = build_w2i(TRAIN_X_PATH, w2c, unk='unk')
    train_x, train_y = build_dataset(TRAIN_X_PATH, TRAIN_Y_PATH, w2i, unk='unk')
    valid_x, valid_y = build_dataset(VALID_X_PATH, VALID_Y_PATH, w2i, unk='unk')

    train_x, train_y = sort_data_by_length(train_x, train_y)
    valid_x, valid_y = sort_data_by_length(valid_x, valid_y)
    print(valid_x)
    VOCAB_SIZE = len(w2i)
    print('VOCAB_SIZE:', VOCAB_SIZE)

    V_init = init_V(w2v, w2i)
    train_iterator, valid_iterator= data.BucketIterator.splits((train_x, valid_x), batch_size = BATCH_SIZE, device = device)

    with open(os.path.join(RESULTS_DIR, './w2i.dump'), 'wb') as f_w2i, open(os.path.join(RESULTS_DIR, './i2w.dump'), 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

    # Build model =================================================================================
 


    model=CNN(VOCAB_SIZE, EMB_DIM, NUM_FIL, WIN_SIZES, OUT_DIM, 
                 DROPOUT_PROB, len(w2i))


    # Train model ================================================================================
    pretrained_embeddings = torch.tensor(V_init)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[len(w2i)-3] = torch.zeros(EMB_DIM)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)    
    criterion = criterion.to(device)
    n_batches_train = math.ceil(len(train_x)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_x)/BATCH_SIZE)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

if __name__ == '__main__':
    main()
>>>>>>> fabc6c1c81f5803acd7fe45035923e5861330d9a
