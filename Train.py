import numpy as np
from Dataset import Dataset
from Loss import TripletLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from Encoder import ResNetEncoder
from Utils import get_preprocessing, split_dataset, calc_euclidean
from matplotlib import pyplot
import torch
import torch.nn.functional as F
import random
import argparse


def run_epoch(data_loader, model, optimizer, loss_func, description, training=True):
    running_loss = []
    pos_euclidean_dist = []
    neg_euclidean_dist = []
    pos_cosine_sim = []
    neg_cosine_sim = []
    for step, (batch_a, batch_p, batch_n) in enumerate(
            tqdm(data_loader, ncols=100, desc=description, leave=True)):
        anchor_img = batch_a.to(device)
        positive_img = batch_p.to(device)
        negative_img = batch_n.to(device)  # must be (1. nn output, 2. target)
        if training:
            optimizer.zero_grad()  # clear gradients for next train
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)
        loss = loss_func(anchor_out, positive_out, negative_out)
        if training:
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        pos_euclidean_dist.append(calc_euclidean(anchor_out, positive_out).cpu().detach().numpy())
        neg_euclidean_dist.append(calc_euclidean(anchor_out, negative_out).cpu().detach().numpy())
        pos_cosine_sim.append(F.cosine_similarity(anchor_out, positive_out).cpu().detach().numpy())
        neg_cosine_sim.append(F.cosine_similarity(anchor_out, negative_out).cpu().detach().numpy())
        running_loss.append(float(loss))
    return (np.concatenate(pos_euclidean_dist).ravel().tolist(), np.concatenate(neg_euclidean_dist).ravel().tolist(),
            np.concatenate(pos_cosine_sim).ravel().tolist(), np.concatenate(neg_cosine_sim).ravel().tolist(),
            sum(running_loss) / len(running_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_dir", default='./Dataset/Extracted Faces/Extracted Faces',
                        help="Path to extracted faces training data directory")
    parser.add_argument("-m", "--model_path", default='./trained_model', help="Path to model (save/load)")
    parser.add_argument("-f", "--feat_extract", default='resnet50',
                        help="Encoder feature extract, options: 'resnet18', 'resnet34', 'resnet50'")
    parser.add_argument("-s", "--sim_metric", default='Euclidean',
                        help="Similarity metric, options: 'Euclidean', 'Cosine'")
    parser.add_argument("-e", "--epochs", default=1, type=int, help="Number of epochs for training")
    parser.add_argument("-b", "--batch", default=5, type=int, help="Batch size")
    parser.add_argument("-r", "--learning_rate", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("-t", "--plot", default=True, type=bool, help="Plot learning trends")
    args = vars(parser.parse_args())

    random.seed(5)
    np.random.seed(5)

    data_dir = args['data_dir']
    train_list, test_list = split_dataset(data_dir, split=0.9)

    mean = [0.485, 0.456, 0.406]  # imagenet mean values
    std = [0.229, 0.224, 0.225]  # imagenet std values

    train_dataset = Dataset(data_dir, train_list, preprocessing=get_preprocessing(mean, std))
    train_loader = DataLoader(train_dataset, batch_size=args['batch'], shuffle=True, num_workers=2)

    test_dataset = Dataset(data_dir, test_list, preprocessing=get_preprocessing(mean, std))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    epochs = args['epochs']
    learning_rate = args['learning_rate']
    path_to_model = args['model_path']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if args['feat_extract'] == 'resnet18':
        encoder = ResNetEncoder(feature_extractor='resnet18').to(device)
    elif args['feat_extract'] == 'resnet34':
        encoder = ResNetEncoder(feature_extractor='resnet34').to(device)
    else:
        encoder = ResNetEncoder().to(device)

    optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    loss_function = TripletLoss(metric=args['sim_metric'])

    try:
        encoder.load_state_dict(torch.load(path_to_model))
        print(f"{path_to_model} valid. state dict loaded")
    except Exception as e:
        print(e)
        print(f"{path_to_model} not valid. state dict not loaded. training from scratch.")

    running_loss = []
    pos_euclidean_dist = []
    neg_euclidean_dist = []
    pos_cosine_sim = []
    neg_cosine_sim = []
    for epoch in range(epochs):
        encoder.train()
        desc = "Epoch " + str(epoch + 1) + "/" + str(epochs)
        epoch_pos_euclidean, epoch_neg_euclidean, epoch_pos_cosine, epoch_neg_cosine, epoch_running_loss = run_epoch(
            data_loader=train_loader, model=encoder, optimizer=optim, loss_func=loss_function,
            description=desc, training=True)

        pos_euclidean_dist.append(epoch_pos_euclidean)
        neg_euclidean_dist.append(epoch_neg_euclidean)
        pos_cosine_sim.append(epoch_pos_cosine)
        neg_cosine_sim.append(epoch_neg_cosine)
        running_loss.append(epoch_running_loss)
        print('Epoch: {} / {} — Loss: {:.4f}'.format(epoch + 1, epochs,
                                                     running_loss[-1]),
              " | Positive Euclidean Distance: ",
              "{:.2f}".format(sum(pos_euclidean_dist[-1]) / len(pos_euclidean_dist[-1])),
              " | Negative Euclidean Distance: ",
              "{:.2f}".format(sum(neg_euclidean_dist[-1]) / len(neg_euclidean_dist[-1])),
              " | Positive Cosine Similarity: ", "{:.2f}".format(sum(pos_cosine_sim[-1]) / len(pos_cosine_sim[-1])),
              " | Negative Cosine Similarity: ", "{:.2f}".format(sum(neg_cosine_sim[-1]) / len(neg_cosine_sim[-1]))
              )

    torch.save(encoder.state_dict(), path_to_model)
    encoder.eval()
    with torch.no_grad():
        test_pos_euclidean, test_neg_euclidean, test_pos_cosine, test_neg_cosine, test_running_loss = run_epoch(
            data_loader=test_loader, model=encoder, optimizer=optim, loss_func=loss_function,
            description='Test', training=False)

        print("Test loss: ", "{:.2f}".format(test_running_loss),
              " | Positive Euclidean Distance: ", "{:.2f}".format(sum(test_pos_euclidean) / len(test_pos_euclidean)),
              " | Negative Euclidean Distance: ", "{:.2f}".format(sum(test_neg_euclidean) / len(test_neg_euclidean)),
              " | Positive Cosine Similarity: ", "{:.2f}".format(sum(test_pos_cosine) / len(test_pos_cosine)),
              " | Negative Cosine Similarity: ", "{:.2f}".format(sum(test_neg_cosine) / len(test_neg_cosine))
              )
    if args['plot']:
        figure, axis = pyplot.subplots(3, 1)
        axis[0].plot(np.arange(1, len(running_loss) + 1), running_loss, label='Train')
        axis[0].set_title('Train Loss')
        axis[0].set(xlabel='Epoch', ylabel='Loss')

        bins = np.linspace(-1, 1, 100)
        axis[1].hist(np.array(test_pos_cosine).squeeze(), bins, alpha=0.5, label='Positive samples')
        axis[1].hist(np.array(test_neg_cosine).squeeze(), bins, alpha=0.5, label='Negative samples')
        axis[1].legend(loc='upper right')
        axis[1].set_title('Cosine Similarity')

        bins = np.linspace(0, np.max(np.array(test_pos_euclidean).squeeze()), 100)
        axis[2].hist(np.array(test_pos_euclidean).squeeze(), bins, alpha=0.5, label='Positive samples')
        axis[2].hist(np.array(test_neg_euclidean).squeeze(), bins, alpha=0.5, label='Negative samples')
        axis[2].legend(loc='upper right')
        axis[2].set_title('Euclidean Distance')
        pyplot.show()
