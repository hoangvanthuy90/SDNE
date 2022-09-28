import torch
import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader
import utils
from data import dataset
from models import model
import numpy as np

if __name__ == '__main__':

    G, Adj, Node =  dataset.Read_graph("cora_edgelist.txt")
    model = model.MNN(Node, 1000, 128, 0.5, 1e-2)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    Data = dataset.Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=100, shuffle=True, )
    device = torch.device( "cpu")
    model = model.to(device)
    model.train()
    for epoch in range(1, 10):
        loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
        for index in Data:
            adj_batch = Adj[index]
            adj_mat = adj_batch[:, index]
            b_mat = torch.ones_like(adj_batch)
            b_mat[adj_batch != 0] = 5

            opt.zero_grad()
            L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
            L_reg = 0
            for param in model.parameters():
                L_reg += 1e-5 * torch.sum(torch.abs(param)) + 1e-4 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            opt.step()
            loss_sum += Loss
            loss_L1 += L_1st
            loss_L2 += L_2nd
            loss_reg += L_reg
        scheduler.step(epoch)
        # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
        print("loss for epoch %d is:" %epoch)
        print("loss_sum is %f" %loss_sum)
        print("loss_L1 is %f" %loss_L1)
        print("loss_L2 is %f" %loss_L2)
        print("loss_reg is %f" %loss_reg)
    model.eval()
    embedding = model.savector(Adj)
    outVec = embedding.detach().numpy()
    np.savetxt("Vec.emb", outVec)