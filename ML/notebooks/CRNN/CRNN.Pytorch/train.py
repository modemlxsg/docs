import torch
import yaml
import os
from dataset import SynthTextDataset
from model_new import CRNN
from utils import collate_pad, decode, convert2str
import logging

# config
config_file = open('config.yaml', 'r', encoding='utf-8')
config = config_file.read()
config_file.close()
config = yaml.full_load(config)

# dict
lexicon = [x for x in config['lexicon']['chars']]
print(lexicon)

# dataset
train_dataset = SynthTextDataset(config, mode='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True,
                                                drop_last=False,
                                                num_workers=0,
                                                collate_fn=collate_pad)

val_dataset = SynthTextDataset(config, mode='val')
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=32,
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            collate_fn=collate_pad)


# model
imgH = config['crnn']['imgH']
nc = config['crnn']['nc']
nClass = config['crnn']['nClass']
nh = config['crnn']['nh']
blank_index = config['lexicon']['blank']
print(blank_index)
# model = CRNN(imgH, nc, nClass, nh)
model = CRNN(nClass)
criterion = torch.nn.CTCLoss(blank=blank_index, reduction='none', zero_infinity = True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# training
epochs = 20
for epoch in range(epochs):

    # train
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    for b, (img, label, label_encode) in enumerate(train_dataloader):
        
        optimizer.zero_grad()
        output = model(img)
        
        input_length = torch.full(size=(output.shape[1],), fill_value=output.shape[0], dtype=torch.long)

        target_length = []
        for la in label:
            target_length.append(len(la))
        target_length = torch.tensor(target_length, dtype=torch.long)

        loss = criterion(output, label_encode, input_length, target_length)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, batch : {b} , loss : {loss}")
    scheduler.step()

    if not os.path.exists('./checkpoints'):
        os.mkdir("./checkpoints")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f"./checkpoints/checkpoint_{epoch}.pth")

    # val
    total = 0
    postive = 0
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    for b, (img, label, label_encode) in enumerate(val_dataloader):
        batch_size = img.shape[0]
        total += batch_size

        output = model(img)
        assert output.shape[1] == batch_size

        out = output.detach().numpy()
        decoded = decode(out)  # [batch, max_length]
        res = convert2str(decoded, lexicon)
        print(out, out.shape)
        print(f"DECODED : {decoded}")
        print(label_encode,"\n", res)
        
        for i in range(len(decoded)):
            if decoded[i] == label_encode[i]:
                postive += 1
    
    print(f'Accuracy : {postive/total}')    






    





     