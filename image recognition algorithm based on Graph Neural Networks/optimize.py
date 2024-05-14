import torch
from data.load_data import load_data
from models.gcn import GCN
from train import train
from evaluate import evaluate

def optimize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = load_data(train=True)
    model = GCN(in_channels=1, hidden_channels=128, out_channels=10)  # 增加隐藏层维度
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, loss_fn)
    torch.save(model.state_dict(), 'optimized_model.pth')

    test_loader = load_data(train=False)
    model.load_state_dict(torch.load('optimized_model.pth'))
    accuracy = evaluate(model, test_loader)
    print(f'Optimized Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    optimize()
