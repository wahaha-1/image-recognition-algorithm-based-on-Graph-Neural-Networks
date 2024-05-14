import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import torch
import openai
from data.load_data import load_data
from models.gcn import GCN
from train import train, generate_image_description
from evaluate import evaluate
from optimize import optimize
from predict import predict

# 设置OpenAI API密钥
openai.api_key = 'your-api-key-here'

class GNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GNN Image Recognition")

        self.dataset_choice = tk.StringVar(value="own")

        self.create_widgets()

    def create_widgets(self):
        # 创建选择框
        self.own_data_radio = tk.Radiobutton(self.root, text="Use Own Dataset", variable=self.dataset_choice, value="own")
        self.own_data_radio.pack()
        
        self.chatgpt_data_radio = tk.Radiobutton(self.root, text="Use ChatGPT Generated Data", variable=self.dataset_choice, value="chatgpt")
        self.chatgpt_data_radio.pack()

        # 创建按钮
        self.train_button = tk.Button(self.root, text="Train", command=self.train_model)
        self.train_button.pack()

        self.import_button = tk.Button(self.root, text="Import Data", command=self.import_data)
        self.import_button.pack()

        self.evaluate_button = tk.Button(self.root, text="Evaluate", command=self.evaluate_model)
        self.evaluate_button.pack()

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_model)
        self.predict_button.pack()

        # 创建文本框以显示输出
        self.output_text = scrolledtext.ScrolledText(self.root, width=80, height=20)
        self.output_text.pack()

    def append_output(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def train_model(self):
        self.append_output("Starting training...")

        choice = self.dataset_choice.get()
        if choice == "own":
            self.append_output("Using own dataset...")
            self.train_with_own_data()
        elif choice == "chatgpt":
            self.append_output("Using ChatGPT generated data...")
            self.train_with_chatgpt_data()
        else:
            messagebox.showerror("Error", "Invalid dataset choice.")

    def train_with_own_data(self):
        train_loader = load_data(train=True)
        model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        train(model, train_loader, optimizer, loss_fn)
        torch.save(model.state_dict(), 'model.pth')
        self.append_output("Training completed and model saved as 'model.pth'.")

    def train_with_chatgpt_data(self):
        train_loader = load_data(train=True)
        model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        for epoch in range(1, 101):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = loss_fn(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.append_output(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader):.4f}')

            # 调用ChatGPT API生成图像描述并评估模型
            image_description = generate_image_description()
            self.append_output(f'Generated Image Description: {image_description}')

        torch.save(model.state_dict(), 'model.pth')
        self.append_output("Training with ChatGPT completed and model saved as 'model.pth'.")

    def import_data(self):
        self.append_output("Importing data...")
        # 这里可以实现导入数据的逻辑
        self.append_output("Data imported successfully.")

    def evaluate_model(self):
        self.append_output("Evaluating model...")
        test_loader = load_data(train=False)
        model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
        model.load_state_dict(torch.load('model.pth'))
        accuracy = evaluate(model, test_loader)
        self.append_output(f"Model accuracy: {accuracy:.4f}")

    def predict_model(self):
        self.append_output("Predicting using model...")
        test_loader = load_data(train=False)
        model = GCN(in_channels=1, hidden_channels=64, out_channels=10)
        model.load_state_dict(torch.load('model.pth'))
        predictions = predict(model, test_loader)
        self.append_output("Predictions:")
        for pred in predictions:
            self.append_output(str(pred))

if __name__ == '__main__':
    root = tk.Tk()
    app = GNNApp(root)
    root.mainloop()
