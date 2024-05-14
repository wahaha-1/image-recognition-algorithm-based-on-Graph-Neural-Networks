import torch
from data.load_data import load_data
from models.gcn import GCN

def predict(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            predictions.append(pred.cpu().numpy())
    return predictions

if __name__ == '__main__':
    test_loader = load_data(train=False)
    model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
    model.load_state_dict(torch.load('model.pth'))

    predictions = predict(model, test_loader)
    for pred in predictions:
        print(pred)
