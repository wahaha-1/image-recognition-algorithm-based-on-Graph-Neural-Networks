import torch
import openai
from data.load_data import load_data
from models.gcn import GCN

# 设置OpenAI API密钥
openai.api_key = 'your-api-key-here'

def generate_image_description():
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Describe an image from the MNIST dataset.",
        max_tokens=50
    )
    return response.choices[0].text.strip()

def train(model, loader, optimizer, loss_fn, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {total_loss / len(loader):.4f}')

        # 调用ChatGPT API生成图像描述并评估模型
        image_description = generate_image_description()
        print(f'Generated Image Description: {image_description}')

if __name__ == '__main__':
    train_loader = load_data(train=True)
    model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, loss_fn)
    torch.save(model.state_dict(), 'model.pth')
