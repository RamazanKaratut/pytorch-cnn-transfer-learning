import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import warnings

# Her türlü sinir bozucu terminal uyarısını tamamen susturur
warnings.filterwarnings("ignore")

# --- 1. Veri Hazırlama (ResNet50 için 224x224) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Demo amaçlı hızlı sürmesi için train setinin sadece bir kısmını kullanabilirsin, ama tam set bırakıyorum.
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# ResNet50 VRAM'i hızlı doldurur, batch_size'ı 64'e düşürdük.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# --- 2. Model Ayarlama Fonksiyonu ---
def get_resnet50(mode="feature_extraction", num_classes=10):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    if mode == "feature_extraction":
        # Ağın tamamını dondur
        for param in model.parameters():
            param.requires_grad = False
            
    # Son katmanı değiştir (Yeni eklenen katman otomatik olarak requires_grad=True olur)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 3. Ortak Eğitim Motoru (Early Stopping Eklendi) ---
def train_model(model, optimizer, criterion, device, epochs, name, patience=3):
    print(f"\n--- {name} Eğitimi Başlıyor ---")
    model.to(device)
    history = []
    
    best_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        epoch_acc = 100. * correct / total
        history.append(epoch_acc)
        print(f"Epoch {epoch+1:02d}/{epochs} | Test Acc: {epoch_acc:.2f}% | Süre: {(time.time()-start_time):.1f} sn")
        
        # --- EARLY STOPPING KONTROLÜ ---
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"-> [DİKKAT] Early Stopping tetiklendi! {patience} epoch boyunca gelişme olmadı.")
            break
            
    print(f"{name} Eğitimi Tamamlandı. En İyi Başarı: {best_acc:.2f}%")
    return history

# --- 4. Ana Çalıştırma Döngüsü ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    
    # Early stopping olduğu için maksimum epoch sınırını rahatça yüksek tutabiliriz
    epochs = 15 
    criterion = nn.CrossEntropyLoss()
    results = {}

    # DENEY 1: Feature Extraction
    model_fe = get_resnet50(mode="feature_extraction")
    # Sadece 'fc' parametreleri optimize edilir
    optimizer_fe = optim.Adam(model_fe.fc.parameters(), lr=0.001)
    results['Feature Extraction'] = train_model(model_fe, optimizer_fe, criterion, device, epochs, "Feature Extraction", patience=3)

    # DENEY 2: Full Fine-Tuning
    model_ft = get_resnet50(mode="fine_tuning")
    # Tüm ağ optimize edilir ama LR çok küçük tutulur (Örn: 1e-4)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    results['Full Fine-Tuning'] = train_model(model_ft, optimizer_ft, criterion, device, epochs, "Full Fine-Tuning", patience=3)

    # Grafik Çizimi
    plt.figure(figsize=(10, 5))
    for name, acc_list in results.items():
        # X eksenini her modelin kendi çalıştığı epoch sayısına göre ayarla
        plt.plot(range(1, len(acc_list) + 1), acc_list, label=f"{name} (Final: {acc_list[-1]:.2f}%)", marker='o')
        
    plt.title('Transfer Learning: Feature Extraction vs Full Fine-Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('hw2_transfer_learning_result.png')
    print("\nGrafik 'hw2_transfer_learning_result.png' olarak kaydedildi.")

if __name__ == '__main__':
    main()