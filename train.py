import torch
from BucketEhingerDataset import BucketEhingerDataset
from EhingerDataset import EhingerDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from AttnDecoderRNN import AttnDecoderRNN
import torch.nn.functional as F
import sys
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)
batch_size = 32


train_set = BucketEhingerDataset(csv_file='ehinger_dataset.csv',
                                   root_dir='Ehinger',
                                   batch_size=batch_size,
                                   transform=transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                           ]))


eval_set = EhingerDataset(csv_file='ehinger_dataset_eval.csv',
                                   root_dir='Ehinger',
                                   batch_size=1,
                                   transform=transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                   ]))

#Model param
hidden_size = 512
input_size = 3
output_size = 3
dropout = 0.0

#Training param
num_iter = 1e10
w = train_set.get_bucket_weights()
w = w/sum(w)
num_iter_per_epoch = int(train_set.__len__()/batch_size)

#log
epoch_loss = 0
best_epoch_loss = sys.maxsize
early_stop = 5
no_improvement = 0



RNN = AttnDecoderRNN(input_size, hidden_size, output_size, dropout_p=dropout, att_len=196, att_size=1024).to(device)
opt = torch.optim.Adamax(lr=2e-3, params=filter(lambda p: p.requires_grad, RNN.parameters()))

for it in range(1, int(num_iter)):
    #get batch
    bucket_id = np.random.choice(len(w), p=w)
    x = train_set.__getitem__(bucket_id)

    landmarks = x["landmarks"].to(device)
    features  = x["features"].to(device)
    it_batch_size = x["batch_size"]
    seq_len = train_set.bucket_dict[bucket_id]

    # transform to (timestep, batch_size, features)
    landmarks = landmarks.permute(1,0,2)

    #run rnn
    rnn_outputs    = torch.zeros(seq_len, it_batch_size, 3, dtype=torch.float32, device=device)
    decoder_hidden = RNN.initHidden(it_batch_size)
    rnn_input      = RNN.initInput(it_batch_size)

    loss=0
    for ri in range(seq_len):
        rnn_output, decoder_hidden, attn_weights = RNN(
            rnn_input, decoder_hidden, features)
        # rnn_input = landmarks[ri]  # Teacher forcing
        rnn_input = rnn_output
        rnn_outputs[ri] = rnn_output
        loss += F.mse_loss(rnn_output, landmarks[ri])

    # Back prop.
    opt.zero_grad()
    loss.backward()

    # Clip gradients when they are getting too large
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, RNN.parameters()), 1.0)

    # Update weights
    opt.step()

    if it % 30 == 0:
        print("Iter",it,": batch loss %.2f" % loss.item())
    epoch_loss += loss.item()

    if it % num_iter_per_epoch == 0:
        epoch_loss = epoch_loss/num_iter_per_epoch
        if epoch_loss < best_epoch_loss :
            print("@New best epoch loss: %.2f" % epoch_loss)
            best_epoch_loss = epoch_loss
            no_improvement = 0


            ###################
            ####doing eval on one item
            ###################
            print("@@@@@@@@@@@@@Evaluating a sample")
            ind = random.randint(0, eval_set.__len__()-1)
            x = eval_set.__getitem__(ind)
            print("GT:",x["landmarks"])
            RNN.eval()
            landmarks = x["landmarks"].to(device).unsqueeze(0)
            features = x["features"].to(device).unsqueeze(0)
            landmarks = landmarks.permute(1, 0, 2)
            seq_len = landmarks.size(0)

            # run rnn
            rnn_outputs = torch.zeros(seq_len, 1, 3, dtype=torch.float32, device=device)
            decoder_hidden = RNN.initHidden(1)
            rnn_input = RNN.initInput(1)

            loss = 0
            for ri in range(seq_len):
                rnn_output, decoder_hidden, attn_weights = RNN(
                    rnn_input, decoder_hidden, features)
                # rnn_input = landmarks[ri]  # Teacher forcing
                rnn_input = rnn_output
                rnn_outputs[ri] = rnn_output

            rnn_outputs = rnn_outputs.permute(1, 0, 2)
            print("PRED:",rnn_outputs)
            ######
            RNN.train()


        else:
            print("@Current best epoch loss: %.2f" % best_epoch_loss)
            print("@Current epoch loss: %.2f" % epoch_loss)
            print("@No epoch improvements")
            no_improvement += 1
        epoch_loss = 0

    if no_improvement == early_stop:
        print("No improvements for ",early_stop,"epochs, exiting")
        sys.exit()




