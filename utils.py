import os
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle

def extract_features(root_dir, keys, layer, transform=None, output_file="EhingerFeatures.pkl"):
    print("Features filename",output_file, "does not exist, extracting...")
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    getattr(model, layer).register_forward_hook(get_activation(layer))
    features = {}

    img_folder = os.path.join(root_dir, "Dataset_STIMULI")
    #keys is a tuple (image, subject)
    images = list(set([t[0] for t in keys]))
    for i in tqdm(range(len(images))):
        img = os.path.join(img_folder, images[i])
        with open(img, 'rb') as f:
            img = Image.open(f).convert('RGB')
            if transform is not None:
                img = transform(img)
        inp = torch.unsqueeze(img, 0).cuda()  # 1,dim,x,x
        model(inp)
        out = activation[layer].squeeze(0)
        out = out.view(out.size(0), -1)
        out = out.permute(1, 0)
        #saving geatures
        features[images[i]] = out.cpu().numpy().astype(np.float32)

    print("Dumping features in filename %s with size %s" % (output_file, str(len(features))))
    pickle.dump(features, open(output_file, 'wb+'))
    del model




