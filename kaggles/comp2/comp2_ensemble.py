import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_SIZE = 288
BATCH_SIZE = 32
DATA_DIR = Path('data/test')
LABELS_FILE = Path('labels.json')
SUBMISSION_FILE = Path('submission.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_CONFIGS = {
    'tiny': 'convnext_tiny.in12k_ft_in1k',
    'small': 'convnext_small.in12k_ft_in1k',
    'base': 'convnext_base.in12k_ft_in1k',
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Настройки командной строки: пути к весам, типы моделей и параметры инференса.
parser = argparse.ArgumentParser(description='Ensemble any number of .pt models and create a submission file')
parser.add_argument('--weights', nargs='+', required=True, help='Paths to model .pt files')
parser.add_argument('--models', nargs='+', required=True, choices=list(MODEL_CONFIGS),
                    help='Architecture for each weight file, in the same order')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('--output', default='submission_ensemble.csv')
parser.add_argument('--blend-weights', nargs='+', type=float, default=None,
                    help='Optional coefficient per model; default is equal weighting')
args = parser.parse_args()

if len(args.weights) != len(args.models):
    parser.error('The number of --weights and --models values must be equal.')

if args.blend_weights is not None:
    if len(args.blend_weights) != len(args.weights):
        parser.error('Provide one --blend-weights value per model.')
    if any(w < 0 for w in args.blend_weights) or sum(args.blend_weights) <= 0:
        parser.error('--blend-weights must be non-negative and have a positive sum.')
    coefficients = np.asarray(args.blend_weights, dtype=np.float32)
    coefficients /= coefficients.sum()
else:
    coefficients = np.full(len(args.weights), 1.0 / len(args.weights), dtype=np.float32)

# Загружаем названия/индексы классов и определяем размер выходного слоя классификатора.
with open(LABELS_FILE, encoding='utf-8') as f:
    labels = json.load(f)
num_classes = len(labels)
submission = pd.read_csv(SUBMISSION_FILE)
if not {'id', 'label'}.issubset(submission.columns):
    raise ValueError('submission.csv must contain id and label columns.')

# Проверяем, что каталог с тестовыми изображениями существует.
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f'Test image directory not found: {DATA_DIR}')
def normalize_id(value):
    return Path(str(value)).stem.lower()
image_map = {
    normalize_id(path.name): path for path in DATA_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
}

# Собираем пути строго в порядке ID из submission.csv.
image_paths = []
for image_id in submission['id']:
    key = normalize_id(image_id)
    if key not in image_map:
        raise FileNotFoundError(f'Test image not found for submission id: {image_id}')
    image_paths.append(image_map[key])
print('Device:', DEVICE)
print('Models in ensemble:', len(args.weights))
print('Images to predict:', len(image_paths))

# Предобработка: изменение размера, перевод в тензор и нормализация ImageNet.
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# Dataset открывает изображение по пути, переводит его в RGB и применяет transform.
class ImageDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return transform(image.convert('RGB'))
loader = DataLoader(ImageDataset(image_paths), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')

# Создаёт модель нужной архитектуры и загружает в неё обученные веса из .pt.
def load_model(weight_path, model_type):
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError(f'Model weights not found: {path}')
    print('Loading:', path, '| architecture:', model_type)
    model = timm.create_model(MODEL_CONFIGS[model_type], pretrained=False,
                              num_classes=num_classes, drop_path_rate=0.1)
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and isinstance(state.get('model'), dict):
        state = state['model']
    elif isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()

# Загружаем все модели; порядок соответствует аргументам --weights и --models.
models = [load_model(path, model_type) for path, model_type in zip(args.weights, args.models)]
probability_batches = [[] for _ in models]
print('\nStarting inference...')
with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
        batch = batch.to(DEVICE, non_blocking=True)
        flipped = torch.flip(batch, dims=[3])
        for model_index, model in enumerate(models):
            probs = (torch.softmax(model(batch), dim=1) +
                     torch.softmax(model(flipped), dim=1)) / 2.0
            probability_batches[model_index].append(probs.cpu().numpy())
        if (batch_index + 1) % 10 == 0:
            print('Processed batches:', batch_index + 1)

model_probabilities = [np.concatenate(chunks, axis=0) for chunks in probability_batches]
ensemble_probabilities = np.zeros_like(model_probabilities[0], dtype=np.float32)
for coefficient, probabilities in zip(coefficients, model_probabilities):
    ensemble_probabilities += coefficient * probabilities
submission['label'] = ensemble_probabilities.argmax(axis=1).astype(int)

# Сохраняем готовый submission-файл без дополнительного столбца индексов.
submission.to_csv(args.output, index=False)
print('\nEnsemble coefficients:')
for path, model_type, coefficient in zip(args.weights, args.models, coefficients):
    print(f'  {path} ({model_type}): {coefficient:.4f}')
print('Submission saved:', args.output)
print('Predictions:', len(submission))
