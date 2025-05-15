import os
import random

import librosa
import numpy as np
import soundfile as sf  # Для создания фиктивных аудио
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import f1_score as sklearn_f1_score  # Для сравнения или если torchmetrics недоступен
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score  # Основная метрика
from torchvision import models


# --- Аугментации (такие же, как в предыдущем ответе) ---
class AddGaussianNoise:
    def __init__(self, min_amplitude=0.001, max_amplitude=0.015):
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, samples):
        noise_amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = torch.randn_like(samples) * noise_amplitude
        return samples + noise


class TimeShift:
    def __init__(self, max_shift_ms, sample_rate):
        self.max_shift_samples = int((max_shift_ms / 1000) * sample_rate)

    def __call__(self, samples):
        shift_samples = random.randint(-self.max_shift_samples, self.max_shift_samples)
        if shift_samples == 0:
            return samples
        shifted_samples = torch.roll(samples, shifts=shift_samples, dims=-1)
        if shift_samples > 0:
            shifted_samples[..., :shift_samples] = 0
        else:
            shifted_samples[..., shift_samples:] = 0
        return shifted_samples


class PitchShift:
    def __init__(self, n_steps_range=(-2, 2), sample_rate=16000):
        self.n_steps_range = n_steps_range
        self.sample_rate = sample_rate

    def __call__(self, samples):
        samples_np = samples.numpy()
        n_steps = random.uniform(self.n_steps_range[0], self.n_steps_range[1])
        # Убедимся, что samples_np одномерный для librosa
        if samples_np.ndim > 1:
            samples_np = samples_np.squeeze(0)
        shifted_samples_np = librosa.effects.pitch_shift(y=samples_np, sr=self.sample_rate, n_steps=n_steps)
        return torch.from_numpy(shifted_samples_np).unsqueeze(0)  # Возвращаем канал


class TimeStretch:
    def __init__(self, rate_range=(0.8, 1.2), sample_rate=16000):  # sample_rate для контекста
        self.rate_range = rate_range

    def __call__(self, samples):
        samples_np = samples.numpy()
        rate = random.uniform(self.rate_range[0], self.rate_range[1])
        if samples_np.ndim > 1:
            samples_np = samples_np.squeeze(0)
        stretched_samples_np = librosa.effects.time_stretch(y=samples_np, rate=rate)
        return torch.from_numpy(stretched_samples_np).unsqueeze(0)  # Возвращаем канал


# --- Класс Датасета для Мульти-лейбл Классификации ---
class HeartSoundMultiLabelDataset(Dataset):
    def __init__(
        self,
        audio_paths,
        multi_labels,
        target_sample_rate=16000,
        max_duration_seconds=60,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        apply_augmentations=False,
        features_for_cnn=True,
    ):
        self.audio_paths = audio_paths
        self.multi_labels = multi_labels  # Список списков/тензоров, e.g., [[1,0,1], [0,1,1]]
        self.target_sample_rate = target_sample_rate
        self.max_samples = target_sample_rate * max_duration_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.apply_augmentations = apply_augmentations
        self.features_for_cnn = features_for_cnn  # True для ResNet/AttentionCNN, False для BiLSTM

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        self.amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        if self.apply_augmentations:
            self.noise_augmentation = AddGaussianNoise()
            self.timeshift_augmentation = TimeShift(max_shift_ms=200, sample_rate=self.target_sample_rate)
            self.pitchshift_augmentation = PitchShift(sample_rate=self.target_sample_rate)
            self.timestretch_augmentation = TimeStretch(sample_rate=self.target_sample_rate)

    def __len__(self):
        return len(self.audio_paths)

    def _load_and_preprocess_audio(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Ошибка загрузки {audio_path}: {e}. Возвращаем тишину.")
            return torch.zeros((1, self.max_samples))

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        return waveform

    def _apply_padding_or_truncation(self, waveform):
        num_channels, current_samples = waveform.shape
        if current_samples > self.max_samples:
            waveform = waveform[:, : self.max_samples]
        elif current_samples < self.max_samples:
            padding_needed = self.max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        # Метки должны быть тензором float для BCEWithLogitsLoss
        label = torch.tensor(self.multi_labels[idx], dtype=torch.float)

        waveform = self._load_and_preprocess_audio(audio_path)

        if self.apply_augmentations:
            if random.random() < 0.5:
                waveform = self.noise_augmentation(waveform)
            if random.random() < 0.5:
                waveform = self.timeshift_augmentation(waveform)
            # Librosa-based augmentations (PitchShift, TimeStretch)
            # Важно: они могут изменить длину, поэтому паддинг/обрезка после них
            if waveform.shape[1] > 0:  # Применять только если есть сигнал
                if random.random() < 0.3:
                    temp_waveform = waveform
                    if random.random() < 0.5:  # PitchShift
                        temp_waveform = self.pitchshift_augmentation(temp_waveform)
                    else:  # TimeStretch
                        temp_waveform = self.timestretch_augmentation(temp_waveform)
                    waveform = temp_waveform

        waveform = self._apply_padding_or_truncation(waveform)
        mel_spectrogram = self.mel_spectrogram_transform(waveform)
        mel_spectrogram_db = self.amplitude_to_db_transform(mel_spectrogram)

        min_val, max_val = mel_spectrogram_db.min(), mel_spectrogram_db.max()
        if max_val > min_val:
            mel_spectrogram_db = (mel_spectrogram_db - min_val) / (max_val - min_val)
        else:
            mel_spectrogram_db = torch.zeros_like(mel_spectrogram_db)

        if self.features_for_cnn:  # Для ResNet / AttentionCNN
            features = mel_spectrogram_db  # (1, n_mels, num_frames)
        else:  # Для BiLSTM
            features = mel_spectrogram_db.squeeze(0).transpose(0, 1)  # (num_frames, n_mels)

        return features, label


# --- Модели (адаптированы для num_classes) ---
class HeartSoundResNet(nn.Module):
    def __init__(self, num_classes):  # num_classes - количество возможных меток
        super().__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if hasattr(models, "ResNet18_Weights") else True
        )
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)  # Возвращает логиты


class HeartSoundBiLSTM(nn.Module):
    def __init__(self, num_classes, input_size=128, hidden_size=128, num_layers=2):  # input_size = n_mels
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 из-за bidirectional

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, num_frames, n_mels)
        h_lstm, _ = self.lstm(x)
        # Берем выход последнего временного шага для классификации
        out = self.fc(h_lstm[:, -1, :])
        return out  # Возвращает логиты


class AttentionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # (B, C, 1, T)
        )
        self.attn = nn.Sequential(nn.Conv1d(64, 1, kernel_size=1), nn.Softmax(dim=-1))  # Вход (B, C, T)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):  # x: (B, 1, H, W) H=n_mels, W=num_frames
        x = self.conv(x)  # (B, 64, 1, W')
        x = x.squeeze(2)  # (B, 64, W')
        attn_weights = self.attn(x)  # (B, 1, W')
        x_attended = (x * attn_weights).sum(dim=-1)  # (B, 64) Weighted sum over time
        return self.fc(x_attended)  # Возвращает логиты


def train_epoch(model, dataloader, criterion, optimizer, device, f1_metric_calculator):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)  # Логиты
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Для F1 score нужны бинарные предсказания
        preds_proba = torch.sigmoid(outputs)
        # preds_binary = (preds_proba > 0.5).float() # Порог 0.5
        # f1_metric_calculator.update(preds_binary, labels.int())
        all_preds.append(preds_proba.detach())  # Собираем вероятности для расчета F1 в конце эпохи
        all_targets.append(labels.detach())

    avg_loss = total_loss / len(dataloader)
    # Расчет F1 для всей эпохи
    epoch_preds = torch.cat(all_preds)
    epoch_targets = torch.cat(all_targets)
    f1_val = f1_metric_calculator(epoch_preds, epoch_targets.int())  # torchmetrics F1 ожидает int таргеты

    return avg_loss, f1_val.item()


def validate_epoch(model, dataloader, criterion, device, f1_metric_calculator):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)  # Логиты
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds_proba = torch.sigmoid(outputs)
            # preds_binary = (preds_proba > 0.5).float()
            # f1_metric_calculator.update(preds_binary, labels.int())
            all_preds.append(preds_proba.detach())
            all_targets.append(labels.detach())

    avg_loss = total_loss / len(dataloader)
    epoch_preds = torch.cat(all_preds)
    epoch_targets = torch.cat(all_targets)
    f1_val = f1_metric_calculator(epoch_preds, epoch_targets.int())

    return avg_loss, f1_val.item()


def predict_sample(model, features_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        features_tensor = features_tensor.unsqueeze(0).to(device)  # Добавляем batch_dim
        logits = model(features_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).cpu().numpy().astype(int)
    return probabilities.cpu().numpy(), predictions


# --- Основной процесс обучения ---
if __name__ == "__main__":
    # --- Функции обучения, валидации и инференса ---
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Используется устройство: {DEVICE}")
    # --- 1. Подготовка данных (фиктивные данные) ---
    TEST_AUDIO_DIR = "test_audio_data_multilabel"
    if not os.path.exists(TEST_AUDIO_DIR):
        os.makedirs(TEST_AUDIO_DIR)

    NUM_CLASSES = 4  # Количество возможных меток для мульти-лейбл
    sample_rate_dummy = 22050
    duration_dummy = 3
    num_dummy_files = 20  # Увеличим для более репрезентативной выборки
    dummy_audio_paths = []
    dummy_multi_labels = []

    for i in range(num_dummy_files):
        num_channels_dummy = 1 if i % 3 == 0 else 2  # Сделаем больше моно
        current_duration = duration_dummy + random.randint(-1, 2)
        current_duration = max(1, current_duration)  # Минимум 1 секунда

        if num_channels_dummy == 1:
            data = np.random.uniform(-0.5, 0.5, size=(sample_rate_dummy * current_duration)).astype("float32")
        else:
            data = np.random.uniform(
                -0.5, 0.5, size=(sample_rate_dummy * current_duration, num_channels_dummy)
            ).astype("float32")

        file_path = os.path.join(TEST_AUDIO_DIR, f"dummy_multilabel_audio_{i}.wav")
        sf.write(file_path, data, sample_rate_dummy)
        dummy_audio_paths.append(file_path)

        # Генерируем случайные мульти-лейбл метки
        # Каждый аудиофайл может иметь от 0 до NUM_CLASSES меток
        num_active_labels = random.randint(0, NUM_CLASSES)  # От 0 до всех меток
        label_vector = [0] * NUM_CLASSES
        if num_active_labels > 0:
            active_indices = random.sample(range(NUM_CLASSES), num_active_labels)
            for idx in active_indices:
                label_vector[idx] = 1
        # Для воспроизводимости можно было бы добавить условие, чтобы хотя бы одна метка была,
        # но для теста подойдет и так.
        if sum(label_vector) == 0 and NUM_CLASSES > 0:  # Гарантируем хотя бы одну метку, если классы есть
            label_vector[random.randint(0, NUM_CLASSES - 1)] = 1

        dummy_multi_labels.append(label_vector)

    print(f"Создано {len(dummy_audio_paths)} фиктивных аудиофайлов.")
    # print(f"Пример меток: {dummy_multi_labels[:5]}")

    # Разделение на train/val (простое)
    split_idx = int(len(dummy_audio_paths) * 0.8)
    train_paths, val_paths = dummy_audio_paths[:split_idx], dummy_audio_paths[split_idx:]
    train_labels, val_labels = dummy_multi_labels[:split_idx], dummy_multi_labels[split_idx:]

    # --- 2. Параметры датасета и даталоадеров ---
    TARGET_SR = 16000
    MAX_DURATION_SEC = 10
    N_MELS = 64  # Уменьшим для скорости на фиктивных данных
    N_FFT = 1024
    HOP_LENGTH = 512
    BATCH_SIZE = 4  # Уменьшим для маленького датасета

    # F1 Score калькулятор (Multi-label)
    # Важно: num_labels=NUM_CLASSES, threshold=0.5 (или можно настроить)
    # average='weighted' для взвешенного F1
    f1_calculator = F1Score(task="multilabel", num_labels=NUM_CLASSES, threshold=0.5, average="weighted").to(DEVICE)

    # --- 3. Обучение моделей ---
    models_to_train = {
        "ResNet18": HeartSoundResNet(num_classes=NUM_CLASSES),
        "BiLSTM": HeartSoundBiLSTM(num_classes=NUM_CLASSES, input_size=N_MELS),  # input_size = n_mels
        "AttentionCNN": AttentionCNN(num_classes=NUM_CLASSES),
    }

    NUM_EPOCHS = 3  # Мало эпох для быстрого теста
    LEARNING_RATE = 0.001

    for model_name, model_instance in models_to_train.items():
        print(f"\n--- Обучение модели: {model_name} ---")
        model = model_instance.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()  # Подходит для мульти-лейбл

        # Определяем, какой формат признаков нужен модели
        features_for_cnn_model = True if model_name in ["ResNet18", "AttentionCNN"] else False

        train_dataset = HeartSoundMultiLabelDataset(
            train_paths,
            train_labels,
            TARGET_SR,
            MAX_DURATION_SEC,
            N_MELS,
            N_FFT,
            HOP_LENGTH,
            apply_augmentations=True,
            features_for_cnn=features_for_cnn_model,
        )
        val_dataset = HeartSoundMultiLabelDataset(
            val_paths,
            val_labels,
            TARGET_SR,
            MAX_DURATION_SEC,
            N_MELS,
            N_FFT,
            HOP_LENGTH,
            apply_augmentations=False,
            features_for_cnn=features_for_cnn_model,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )  # num_workers=0 для простоты
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        best_val_f1 = -1

        for epoch in range(NUM_EPOCHS):
            f1_calculator.reset()  # Сбрасываем состояние метрики для каждого вызова train/val
            train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, f1_calculator)
            f1_calculator.reset()
            val_loss, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE, f1_calculator)

            print(f"Эпоха {epoch+1}/{NUM_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Weighted F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f},   Val Weighted F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # torch.save(model.state_dict(), f"{model_name}_best_multilabel.pth")
                print(f"  Сохранена новая лучшая модель с Val F1: {best_val_f1:.4f}")

        # --- 4. Инференс на одном примере из валидационного набора ---
        print(f"\n--- Инференс для модели: {model_name} ---")
        if len(val_dataset) > 0:
            sample_idx = 0
            features_sample, label_sample = val_dataset[sample_idx]

            # Убедимся, что features_sample имеет правильную форму для модели
            # Dataset возвращает (C, H, W) или (Seq, Feat)
            # Модели ожидают батч, поэтому predict_sample добавляет batch_dim

            probabilities, predictions = predict_sample(model, features_sample, DEVICE)
            print(f"Пример {sample_idx} из валидационного набора:")
            print(f"  Истинные метки:    {label_sample.numpy().astype(int)}")
            print(f"  Вероятности:       {[f'{p:.2f}' for p in probabilities.squeeze()]}")
            print(f"  Предсказанные метки: {predictions.squeeze()}")
        else:
            print("Валидационный набор пуст, инференс пропущен.")

    # Очистка фиктивных файлов (раскомментировать при необходимости)
    # for p in dummy_audio_paths:
    #     if os.path.exists(p):
    #         os.remove(p)
    # if os.path.exists(TEST_AUDIO_DIR):
    #     os.rmdir(TEST_AUDIO_DIR)
    print(f"\nНе забудьте удалить тестовую директорию: {TEST_AUDIO_DIR}")
