import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 50)
print("INICIANDO DIAGNÓSTICO Y ENTRENAMIENTO DE PRUEBA")
print("=" * 50)

# 1. Diagnóstico del Sistema en PyTorch
print(f"Versión de Python: {sys.version}")
print(f"Versión de PyTorch: {torch.__version__}")

# Intentar cargar torch-directml
try:
    import torch_directml
    dml_available = True
    device = torch_directml.device()
    print("DirectML detectado correctamente!")
    print(f"Dispositivo activo: {device}")
except Exception as e:
    dml_available = False
    device = torch.device("cpu")
    print(f"DirectML no disponible. Error: {e}")
    print("Se usará la CPU para el entrenamiento.")

print("-" * 50)

# 2. Generación de un Dataset Ficticio para Clasificación de Texto
print("Generando dataset de prueba...")
class DummyTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Frases simples de prueba (Clasificación de Sentimientos)
texts = [
    "Me encanta este modelo de inteligencia artificial, es increíble.",
    "El rendimiento del entrenamiento en esta máquina es fantástico.",
    "Este hardware funciona muy bien con aceleración de hardware.",
    "Tengo problemas para afinar modelos grandes en mi equipo antiguo.",
    "El entrenamiento es lento y consume demasiados recursos sin aceleración.",
    "No me gusta cómo funciona esta librería, tiene muchos fallos."
]
labels = [1, 1, 1, 0, 0, 0] # 1: Positivo, 0: Negativo

# 3. Cargar Modelo y Tokenizer ultra-pequeño
model_name = "google/bert_uncased_L-2_H-128_A-2" # Un modelo BERT minúsculo para descarga rápida (aprox. 17 MB)
print(f"Descargando y cargando modelo '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Mover el modelo al dispositivo (DirectML o CPU)
model = model.to(device)
print(f"Modelo cargado y transferido al dispositivo: {device}")

# Preparar DataLoader
dataset = DummyTextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. Configurar optimizador y función de pérdida
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# 5. Bucle de Entrenamiento de Prueba (3 Épocas)
print("-" * 50)
print("Iniciando bucle de entrenamiento...")
start_time = time.time()

model.train()
for epoch in range(3):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        # Mover datos al dispositivo acelerado
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_device = batch["label"].to(device)

        # Paso hacia adelante (Forward pass)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Calcular pérdida
        loss = criterion(logits, labels_device)
        
        # Paso hacia atrás (Backward pass)
        loss.backward()
        
        # Optimización
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Época {epoch+1}/3 - Pérdida Media: {epoch_loss / len(dataloader):.4f}")

end_time = time.time()
print("-" * 50)
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos!")
print("¡El entorno de afinamiento está configurado y funcionando perfectamente!")
print("=" * 50)
