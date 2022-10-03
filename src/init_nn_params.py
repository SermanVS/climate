import torch
from SimpleNN import SimpleNN
def init_nn_params(hyperparameters):

    if (len(hyperparameters) == 4):
        w = torch.empty(size=(), dtype=torch.float)
        a = torch.empty(size=(), dtype=torch.float)
        b = torch.empty(size=(), dtype=torch.float)
        c = torch.empty(size=(), dtype=torch.float)

        nn.init.constant_(w, val=0.5) # weight init
        nn.init.constant_(a, val=1)
        nn.init.constant_(b, val=50)
        nn.init.constant_(c, val=2)

        cnn = SimpleNN(w, a, b, c)
    elif (len(hyperparameters) == 3):
        d = torch.empty(size=(), dtype=torch.float)
        e = torch.empty(size=(), dtype=torch.float)
        c = torch.empty(size=(), dtype=torch.float)

        nn.init.constant_(d, val=25)
        nn.init.constant_(e, val=50)
        nn.init.constant_(c, val=2)

        cnn = SimpleNN(d, e, c)
    

    y = torch.tensor(list(labels_train))
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
    class_weights=torch.tensor(class_weights, dtype=torch.double)

    loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0], reduction='mean') 
    optimizer = optim.Adam(cnn.parameters(), lr = 0.0001)  
    return cnn, loss_func, optimizer