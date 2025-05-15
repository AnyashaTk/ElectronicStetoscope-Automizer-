import random

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

# --- Аугментации ---


class AddGaussianNoise:
    """
    Добавляет Гауссовский шум к аудиосигналу.
    """

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015):
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, samples):
        noise_amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        noise = torch.randn_like(samples) * noise_amplitude
        return samples + noise


class TimeShift:
    """
    Сдвигает аудиосигнал во времени.
    """

    def __init__(self, max_shift_ms, sample_rate):
        self.max_shift_samples = int((max_shift_ms / 1000) * sample_rate)

    def __call__(self, samples):
        shift_samples = random.randint(-self.max_shift_samples, self.max_shift_samples)
        if shift_samples == 0:
            return samples
        shifted_samples = torch.roll(samples, shifts=shift_samples, dims=-1)
        # Заполняем тишиной появившиеся пустые места
        if shift_samples > 0:
            shifted_samples[..., :shift_samples] = 0
        else:
            shifted_samples[..., shift_samples:] = 0
        return shifted_samples


class PitchShift:
    """
    Изменяет высоту тона аудиосигнала.
    Использует librosa, так как в torchaudio нет прямого аналога с такой же гибкостью.
    """

    def __init__(self, n_steps_range=(-2, 2), sample_rate=16000):
        self.n_steps_range = n_steps_range
        self.sample_rate = sample_rate

    def __call__(self, samples):
        # Конвертируем тензор PyTorch в NumPy массив
        samples_np = samples.numpy()
        n_steps = random.uniform(self.n_steps_range[0], self.n_steps_range[1])
        # librosa.effects.pitch_shift работает с NumPy массивами
        shifted_samples_np = librosa.effects.pitch_shift(
            y=samples_np.squeeze(0), sr=self.sample_rate, n_steps=n_steps  # Удаляем размерность канала, если она есть
        )
        # Возвращаем в формат тензора PyTorch и добавляем размерность канала
        return torch.from_numpy(shifted_samples_np).unsqueeze(0)


class TimeStretch:
    """
    Растягивает или сжимает аудиосигнал во времени без изменения высоты тона.
    Использует librosa.
    """

    def __init__(self, rate_range=(0.8, 1.2), sample_rate=16000):
        self.rate_range = rate_range
        self.sample_rate = (
            sample_rate  # Не используется напрямую в librosa.effects.time_stretch, но полезно для контекста
        )

    def __call__(self, samples):
        # Конвертируем тензор PyTorch в NumPy массив
        samples_np = samples.numpy()
        rate = random.uniform(self.rate_range[0], self.rate_range[1])
        # librosa.effects.time_stretch работает с NumPy массивами
        stretched_samples_np = librosa.effects.time_stretch(
            y=samples_np.squeeze(0), rate=rate  # Удаляем размерность канала
        )
        # Возвращаем в формат тензора PyTorch и добавляем размерность канала
        return torch.from_numpy(stretched_samples_np).unsqueeze(0)


# --- Класс Датасета ---


class HeartSoundDataset(Dataset):
    def __init__(
        self,
        audio_paths,
        labels,
        target_sample_rate=16000,
        max_duration_seconds=60,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        apply_augmentations=False,
        resnet_mode=True,
    ):
        """
        Инициализация датасета звуков сердца.

        Параметры:
            audio_paths (list): Список путей к аудиофайлам.
            labels (list): Список меток (0 или 1).
            target_sample_rate (int): Целевая частота дискретизации.
            max_duration_seconds (int): Максимальная длительность аудио в секундах (для паддинга/обрезки).
            n_mels (int): Количество Мел-фильтров для спектрограммы.
            n_fft (int): Размер окна FFT.
            hop_length (int): Шаг окна FFT.
            apply_augmentations (bool): Применять ли аугментации.
            resnet_mode (bool): Если True, возвращает спектрограмму с 1 каналом (для ResNet, AttentionCNN).
                               Если False, может возвращать признаки другой формы, если потребуется для LSTM
                               (в данном случае LSTM тоже будет использовать спектрограмму).
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.max_samples = target_sample_rate * max_duration_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.apply_augmentations = apply_augmentations
        self.resnet_mode = resnet_mode  # Используем для определения формы выходных данных

        # --- Инициализация трансформаций для извлечения признаков ---
        # Мел-спектрограмма
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,  # Используем степень 2 для энергетической спектрограммы
        )
        # Преобразование амплитуды в децибелы
        self.amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # --- Инициализация аугментаций (если включены) ---
        if self.apply_augmentations:
            self.augmentations = torch.nn.Sequential(
                # Важно: TimeStretch и PitchShift лучше применять к сырому аудио до спектрограммы
                # Но для простоты и согласованности с другими, которые могут работать на тензорах,
                # мы можем их обернуть. Однако, PitchShift и TimeStretch здесь работают с NumPy.
                # Поэтому их нужно будет применять отдельно перед transform.
            )
            self.noise_augmentation = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01)
            self.timeshift_augmentation = TimeShift(max_shift_ms=200, sample_rate=self.target_sample_rate)
            # Для PitchShift и TimeStretch нужна частота дискретизации,
            # и они возвращают NumPy, поэтому их применение будет чуть отличаться.
            self.pitchshift_augmentation = PitchShift(n_steps_range=(-1, 1), sample_rate=self.target_sample_rate)
            self.timestretch_augmentation = TimeStretch(rate_range=(0.9, 1.1), sample_rate=self.target_sample_rate)

    def __len__(self):
        return len(self.audio_paths)

    def _load_and_preprocess_audio(self, audio_path):
        """
        Загружает аудио, преобразует в моно, ресемплирует.
        """
        try:
            # Загрузка аудио с помощью torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Ошибка загрузки файла {audio_path}: {e}")
            # Возвращаем тензор нулей в случае ошибки, чтобы обучение не прерывалось
            # или можно пропустить этот файл (требует доработки логики в __getitem__)
            return torch.zeros((1, self.max_samples))

        # 1. Преобразование в моно (если стерео, берем среднее по каналам)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Ресемплинг до target_sample_rate
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # 3. Нормализация громкости (простой вариант - деление на максимальное абсолютное значение)
        # Это помогает справиться с разной громкостью записей
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        return waveform

    def _apply_padding_or_truncation(self, waveform):
        """
        Применяет паддинг или обрезку к аудиосигналу до self.max_samples.
        """
        num_channels, current_samples = waveform.shape

        if current_samples > self.max_samples:
            # Обрезка: берем начало аудио
            waveform = waveform[:, : self.max_samples]
        elif current_samples < self.max_samples:
            # Паддинг: добавляем тишину в конец
            padding_needed = self.max_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # 1. Загрузка и базовое предобработка аудио
        waveform = self._load_and_preprocess_audio(audio_path)  # (1, num_samples)

        # 2. Аугментации (если включены)
        # Некоторые аугментации лучше применять к сырому аудио сигналу
        if self.apply_augmentations:
            # Аугментации, работающие с тензорами PyTorch напрямую
            if random.random() < 0.5:  # Применяем с вероятностью 50%
                waveform = self.noise_augmentation(waveform)
            if random.random() < 0.5:
                waveform = self.timeshift_augmentation(waveform)

            # Аугментации, использующие librosa (требуют конвертации в NumPy и обратно)
            # Применяем их до паддинга/обрезки, так как они могут изменить длину
            waveform.shape[1]

            if random.random() < 0.3:  # Меньшая вероятность для более "сильных" аугментаций
                # Убедимся, что waveform это (1, num_samples) или (num_samples) для librosa
                temp_waveform_for_librosa = waveform
                if temp_waveform_for_librosa.ndim == 2 and temp_waveform_for_librosa.shape[0] == 1:
                    temp_waveform_for_librosa = temp_waveform_for_librosa.squeeze(0)

                if temp_waveform_for_librosa.ndim == 1:  # Должен быть 1D для librosa
                    if random.random() < 0.5:  # PitchShift
                        temp_waveform_for_librosa = self.pitchshift_augmentation(
                            temp_waveform_for_librosa.unsqueeze(0)
                        ).squeeze(
                            0
                        )  # Добавляем и убираем канал
                    else:  # TimeStretch
                        temp_waveform_for_librosa = self.timestretch_augmentation(
                            temp_waveform_for_librosa.unsqueeze(0)
                        ).squeeze(
                            0
                        )  # Добавляем и убираем канал

                    # После TimeStretch длина может измениться, нужно обработать
                    # Простой вариант: если сильно изменилась, отменить или обрезать/дополнить до original_length
                    # Но лучше обрезать/дополнять уже на следующем шаге (_apply_padding_or_truncation)
                    waveform = temp_waveform_for_librosa.unsqueeze(0)  # Возвращаем размерность канала
                else:
                    print(f"Предупреждение: Не удалось применить Pitch/Stretch к waveform с формой {waveform.shape}")

        # 3. Паддинг или обрезка до фиксированной длины
        waveform = self._apply_padding_or_truncation(waveform)  # (1, max_samples)

        # 4. Извлечение Мел-спектрограммы
        # (канал, частота, время) -> (1, n_mels, num_frames)
        mel_spectrogram = self.mel_spectrogram_transform(waveform)

        # 5. Преобразование в децибелы
        mel_spectrogram_db = self.amplitude_to_db_transform(mel_spectrogram)

        # 6. Нормализация спектрограммы (например, к [0, 1] или z-score)
        # Простая нормализация к [0,1] на основе min-max всего датасета (лучше)
        # или на основе текущего семпла (проще, но менее стабильно)
        # Здесь пример нормализации на основе текущего семпла:
        min_val = mel_spectrogram_db.min()
        max_val = mel_spectrogram_db.max()
        if max_val > min_val:
            mel_spectrogram_db = (mel_spectrogram_db - min_val) / (max_val - min_val)
        else:  # Если все значения одинаковы (например, тишина)
            mel_spectrogram_db = torch.zeros_like(mel_spectrogram_db)

        # ResNet и AttentionCNN ожидают вход (batch, channels, height, width)
        # BiLSTM может ожидать (batch, seq_len, features)
        # В нашем случае, для ResNet/AttentionCNN: channels=1, height=n_mels, width=num_frames
        # Для BiLSTM: seq_len=num_frames, features=n_mels (транспонируем спектрограмму)

        if self.resnet_mode:
            # mel_spectrogram_db уже имеет форму (1, n_mels, num_frames), что подходит
            # Conv2d в ResNet (model.conv1) ожидает (batch_size, in_channels, height, width)
            # Наш mel_spectrogram_db: (1, n_mels, time_frames) - это и есть то, что нужно.
            # `in_channels` для ResNet conv1 мы поменяли на 1.
            # `height` будет `n_mels`, `width` будет `num_frames`.
            features = mel_spectrogram_db  # (1, n_mels, num_frames)
        else:  # Для BiLSTM
            # LSTM ожидает (batch, seq_len, input_size)
            # Мы можем транспонировать спектрограмму: (1, n_mels, num_frames) -> (1, num_frames, n_mels)
            # Убираем канал, так как LSTM обычно не работает с "каналами" напрямую в этом контексте.
            features = mel_spectrogram_db.squeeze(0).transpose(0, 1)  # (num_frames, n_mels)
            # `input_size` для BiLSTM будет `n_mels`. `seq_len` будет `num_frames`.

        return features, torch.tensor(label, dtype=torch.long)


# --- Пример использования ---
if __name__ == "__main__":
    # Создадим несколько фиктивных аудиофайлов для теста
    import os

    import soundfile as sf

    TEST_AUDIO_DIR = "test_audio_data"
    if not os.path.exists(TEST_AUDIO_DIR):
        os.makedirs(TEST_AUDIO_DIR)

    sample_rate_dummy = 22050  # Изначальная частота для теста
    duration_dummy = 5  # секунд
    num_dummy_files = 4
    dummy_audio_paths = []
    dummy_labels = []

    for i in range(num_dummy_files):
        # Создаем моно и стерео файлы разной длины
        num_channels_dummy = 1 if i % 2 == 0 else 2
        current_duration = duration_dummy + i * 2  # 5, 7, 9, 11 секунд
        if num_channels_dummy == 1:
            data = np.random.uniform(-0.5, 0.5, size=(sample_rate_dummy * current_duration)).astype("float32")
        else:
            data = np.random.uniform(
                -0.5, 0.5, size=(sample_rate_dummy * current_duration, num_channels_dummy)
            ).astype("float32")

        file_path = os.path.join(TEST_AUDIO_DIR, f"dummy_audio_{i}.wav")
        sf.write(file_path, data, sample_rate_dummy)
        dummy_audio_paths.append(file_path)
        dummy_labels.append(i % 2)  # Простые метки 0 или 1

    print(f"Созданы фиктивные аудио: {dummy_audio_paths}")
    print(f"Метки: {dummy_labels}")

    TARGET_SR = 16000
    MAX_DURATION_SEC = 15  # Максимальная длина для обработки
    N_MELS = 64  # Для примера, ResNet18 не очень глубокий, много мел-компонент может быть избыточно
    N_FFT = 1024
    HOP_LENGTH = 512

    print("\n--- Тестирование для ResNet/AttentionCNN (resnet_mode=True) ---")
    # `resnet_mode=True` - спектрограмма (1, n_mels, num_frames)
    dataset_resnet = HeartSoundDataset(
        audio_paths=dummy_audio_paths,
        labels=dummy_labels,
        target_sample_rate=TARGET_SR,
        max_duration_seconds=MAX_DURATION_SEC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        apply_augmentations=True,
        resnet_mode=True,
    )

    # Проверим один элемент
    features_resnet, label_resnet = dataset_resnet[0]
    print(f"Форма признаков для ResNet/AttentionCNN: {features_resnet.shape}")  # Ожидаем (1, N_MELS, num_frames)
    print(f"Метка: {label_resnet}")

    # Ожидаемое количество фреймов во времени:
    # (MAX_SAMPLES - N_FFT) / HOP_LENGTH + 1  (если center=False для спектрограммы)
    # или MAX_SAMPLES / HOP_LENGTH (если center=True, что torchaudio делает по умолчанию для MelSpectrogram)
    expected_num_frames = (TARGET_SR * MAX_DURATION_SEC) // HOP_LENGTH
    # Torchaudio MelSpectrogram с center=True добавляет паддинг, поэтому количество фреймов будет ceil(max_samples / hop_length)
    # или просто max_samples / hop_length если max_samples кратно hop_length.
    # Точнее, это будет int(self.max_samples / self.hop_length) + 1
    # Учитывая, что torchaudio.transforms.MelSpectrogram по умолчанию использует center=True,
    # паддинг применяется так, что количество фреймов равно floor(waveform_length / hop_length) + 1.
    # Для фиксированной waveform_length = self.max_samples, это будет floor(self.max_samples / self.hop_length) + 1
    # В последней версии torchaudio это может быть просто `int(padded_waveform_length / hop_length) + 1`
    # Для простоты, давайте выведем это значение из формы:
    print(f"Ожидаемое N_MELS: {N_MELS}, Фактическое N_MELS: {features_resnet.shape[1]}")
    print(f"Фактическое количество временных фреймов: {features_resnet.shape[2]}")

    print("\n--- Тестирование для BiLSTM (resnet_mode=False) ---")
    # `resnet_mode=False` - спектрограмма (num_frames, n_mels)
    dataset_bilstm = HeartSoundDataset(
        audio_paths=dummy_audio_paths,
        labels=dummy_labels,
        target_sample_rate=TARGET_SR,
        max_duration_seconds=MAX_DURATION_SEC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        apply_augmentations=True,
        resnet_mode=False,  # Важно для BiLSTM
    )

    features_bilstm, label_bilstm = dataset_bilstm[0]
    print(f"Форма признаков для BiLSTM: {features_bilstm.shape}")  # Ожидаем (num_frames, N_MELS)
    print(f"Метка: {label_bilstm}")
    print(f"Ожидаемое N_MELS (features): {N_MELS}, Фактическое N_MELS: {features_bilstm.shape[1]}")
    print(f"Фактическое количество временных фреймов (seq_len): {features_bilstm.shape[0]}")

    # --- Проверка DataLoader ---
    from torch.utils.data import DataLoader

    print("\n--- Тестирование DataLoader для ResNet/AttentionCNN ---")
    dataloader_resnet = DataLoader(dataset_resnet, batch_size=2, shuffle=True)
    for batch_features, batch_labels in dataloader_resnet:
        print(f"Форма батча признаков (ResNet): {batch_features.shape}")  # Ожидаем (batch_size, 1, N_MELS, num_frames)
        print(f"Форма батча меток (ResNet): {batch_labels.shape}")  # Ожидаем (batch_size)
        break  # Показываем только первый батч

    print("\n--- Тестирование DataLoader для BiLSTM ---")
    # Для BiLSTM, если мы используем batch_first=True, данные должны быть (batch, seq, feature)
    # Наш датасет возвращает (seq, feature). DataLoader соберет их в (batch, seq, feature).
    dataloader_bilstm = DataLoader(dataset_bilstm, batch_size=2, shuffle=True)
    for batch_features, batch_labels in dataloader_bilstm:
        print(f"Форма батча признаков (BiLSTM): {batch_features.shape}")  # Ожидаем (batch_size, num_frames, N_MELS)
        print(f"Форма батча меток (BiLSTM): {batch_labels.shape}")  # Ожидаем (batch_size)
        break  # Показываем только первый батч

    # Удаление временных файлов
    # for p in dummy_audio_paths:
    #     if os.path.exists(p):
    #         os.remove(p)
    # if os.path.exists(TEST_AUDIO_DIR):
    #     os.rmdir(TEST_AUDIO_DIR)
    print(f"\nНе забудьте удалить тестовую директорию: {TEST_AUDIO_DIR}")
