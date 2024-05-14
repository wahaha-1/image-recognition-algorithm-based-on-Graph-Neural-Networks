import torch
from data.load_data import load_data
from models.gcn import GCN

def evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':
    test_loader = load_data(train=False)
    model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
    model.load_state_dict(torch.load('model.pth'))
    accuracy = evaluate(model, test_loader)
    print(f'Accuracy: {accuracy:.4f}')
