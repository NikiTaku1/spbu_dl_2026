# Kaggle №2: Распознавание персонажей One Piece

## 1. Назначение

Пайплайн состоит из двух скриптов:

-   `comp2_train.py` обучает одну модель ConvNeXt (`tiny`, `small`
    или `base`) на размеченных изображениях из `data/train` и сохраняет
    веса в `.pt`.
-   `comp2_ensemble.py` загружает любое число файлов `.pt`,
    предсказывает классы для `data/test`, объединяет вероятности моделей
    и создаёт CSV на основе приложенного `submission.csv`.

В ансамблевом скрипте коэффициенты по умолчанию равны, либо задаются явно
через CLI.

## 2. Структура проекта

Команды запускаются из корневой директории проекта. Ожидаемая структура:

``` text
project/
├── comp2_train.py
├── comp2_ensemble.py
├── labels.json
├── submission.csv
└── data/
    ├── train/
    │   ├── Ace/
    │   │   ├── 001.png
    │   │   └── ...
    │   └── Akainu/
    │       └── ...
    └── test/
        ├── (image_name).jpg
        └── ...
```

### Входные файлы

-   `labels.json` --- словарь числовых индексов и названий классов.
    Скрипт обучения строит обратное соответствие «имя папки → индекс».
-   Был приложен файл `train_annotations.csv` с разметкой картинок по       классам, но структура папки train позволила обучать по названиям подпапок, поэтому файл не использовался.
-   `data/train/<class_name>/` --- обучающие изображения. Поддерживаются
    `.jpg`, `.jpeg`, `.png`.
-   `data/test/` --- изображения для инференса.
-   `submission.csv` --- шаблон с обязательными столбцами `id` и
    `label`. Для каждого `id` ансамбль ищет файл в `data/test` по имени
    без расширения и без учёта регистра.

## 3. Зависимости и устройство

Используются `torch`, `torchvision`, `timm`, `numpy`, `pandas`,
`Pillow`.

``` bash
pip install torch torchvision timm numpy pandas pillow
```

Для CUDA нужно установить совместимые сборки PyTorch и TorchVision. Обучающий скрипт создаёт предобученную модель через `timm` (`pretrained=True`),
поэтому при первом запуске может потребоваться доступ к источнику
предобученных весов.

Оба скрипта выбирают `cuda`, если CUDA доступна, иначе `cpu`. В обучении
AMP включается только на CUDA. Для загрузчиков используется
`num_workers=0`; `pin_memory` включается на CUDA.

## 4. Обучение модели

### Запуск

``` bash
python "comp2_train.py" --model tiny
python "comp2_train.py" --model small
python "comp2_train.py" --model base
```

Допустимы `tiny`, `small`, `base`; по умолчанию используется `tiny`. В
скрипте нет CLI-аргументов для эпох, batch size, learning rate или имени
выходного файла --- эти параметры задаются в коде.

### Конфигурация архитектур

- Tiny: `convnext_tiny.in12k_ft_in1k`, Batch size: 64, Start learning rate: `3e-4`
- Small: `convnext_small.in12k_ft_in1k`, Batch size: 32, Start learning rate: `1.5e-4`
- Base: `convnext_base.in12k_ft_in1k`, Batch size: 16, Start learning rate: `1e-4`

Модель создаётся с `pretrained=True`, `num_classes=len(labels)` и
`drop_path_rate=0.1`.

### Основные настройки
  Параметры, базовое значение, назначение:
- `IMG` - 192, Размер для обучающего crop
- `TEST_IMG` - 288, Размер eval-transform
- `FULL_TRAIN` - `True`, Использовать весь размеченный набор для обучения
- `BALANCED_SAMPLING` - `True`, Взвешенная выборка с балансировкой классов
- `VAL_FRAC` - 0.15, Доля, используемая при формировании потенциального validation split
- `SEED` - 0, Seed Python, NumPy, PyTorch
- `TRAIN_EPOCHS` - 70, Количество эпох
- `HEAD_LR_MULTIPLIER` - 10.0, Множитель LR классификатора относительно backbone
- `weight_decay` - 0.05, AdamW weight decay
- `label_smoothing` - 0.1, Сглаживание меток в CrossEntropyLoss


### Подготовка данных

Скрипт сортирует подпапки `data/train`, собирает пути к изображениям
известных классов и создаёт таблицу `path/label`. Если изображения не
найдены, обучение прекращается с ошибкой. Изображения открываются через
Pillow, переводятся в RGB (палитровые `P` сначала переводятся в RGBA),
затем предварительно уменьшаются методом `thumbnail` до ограничения
сторон `2 * max(IMG, TEST_IMG)` --- 576 пикселей при текущих настройках.
Dataset хранит подготовленные изображения в памяти и применяет
transforms при каждом обращении.

### Train/validation split --- важное замечание

Код рассчитывает validation-индексы отдельно внутри каждого класса:
примерно `VAL_FRAC` примеров, но не больше `len(class)-1`; для класса с
одним изображением validation-пример не выделяется.

При `FULL_TRAIN=True` `tr_df` содержит весь датасет,
`va_df` пуст, а `va_dl=None`. Следовательно, фактический запуск
обучается на всех размеченных изображениях без validation-оценки.
Функции Accuracy и Macro-F1 в файле определены, но основной цикл
обучения их не вызывает. Если вручную установить `FULL_TRAIN=False`,
split будет применён и создастся validation DataLoader, однако отдельный
validation-цикл и вычисление метрик в текущем основном цикле всё равно
не реализованы.

### Аугментации

Обучающий transform последовательно применяет:

1.  `RandomResizedCrop(192, scale=(0.45, 1.0), ratio=(0.8, 1.25))`.
2.  `RandomHorizontalFlip()`.
3.  `TrivialAugmentWide()`.
4.  `RandomRotation(8)`.
5.  `ColorJitter(0.18, 0.18, 0.18, 0.02)`.
6.  `RandomGrayscale(0.02)`.
7.  `ToTensor()`.
8.  `Normalize()` со статистиками ImageNet.
9.  `RandomErasing(p=0.15, scale=(0.02, 0.10))`.

Нормализация: `MEAN=(0.485, 0.456, 0.406)`, `STD=(0.229, 0.224, 0.225)`.
Eval-transform задаёт `Resize((288, 288))`, `ToTensor()` и ту же
нормализацию; при `FULL_TRAIN=True` он не используется обучающим
циклом.

### Балансировка классов

При `BALANCED_SAMPLING=True` вес каждого изображения равен обратному
количеству изображений его класса: `1 / count(class)`.
`WeightedRandomSampler` выбирает `len(tr_df)` элементов с
`replacement=True`, увеличивая шанс появления примеров редких классов.
Если балансировка отключена, используется `shuffle=True`.

### MixUp и CutMix

Функция `mix_batch` выбирает один из трёх вариантов:

-   с вероятностью 0.5 оставляет batch без смешивания;
-   при следующей четверти диапазона применяет MixUp с
    `lambda ~ Beta(0.3, 0.3)`;
-   в оставшемся случае применяет CutMix с начальным
    `lambda ~ Beta(1.0, 1.0)`, заменяя прямоугольную область одного
    изображения областью другого и пересчитывая lambda по фактической
    площади замены.

Loss для смешанного batch:

``` text
loss = lambda * CE(logits, label_a) + (1 - lambda) * CE(logits, label_b)
```

### Оптимизация и расписание learning rate

Используется `AdamW(weight_decay=0.05)`. Параметры делятся на backbone и
classifier head (через `net.get_classifier()`). Максимальный LR головы в
10 раз выше LR backbone. Для групп задаются начальные LR как
соответствующие максимальные LR, делённые на `DIV_FACTOR=25`.

`OneCycleLR` настроен так:

-   `epochs=70`;
-   `steps_per_epoch=len(tr_dl)`;
-   `pct_start=0.15`;
-   `anneal_strategy="cos"`;
-   `div_factor=25`;
-   `final_div_factor=10000`.

Scheduler обновляется на каждом batch. Функция потерь:
`CrossEntropyLoss(label_smoothing=0.1)`. На CUDA используется
`torch.autocast` с `float16` и `GradScaler`; на CPU AMP отключён.

### Шаги одной эпохи

1.  `net.train()`.
2.  Перебор batch из train DataLoader.
3.  Перенос изображений и меток на выбранное устройство.
4.  MixUp/CutMix либо исходный batch.
5.  Forward pass и расчёт смешанного loss.
6.  `zero_grad(set_to_none=True)`.
7.  Backward pass через GradScaler.
8.  Шаг AdamW и обновление GradScaler.
9.  Шаг OneCycleLR после каждого batch.
10. Расчёт среднего loss по batch.
11. Сохранение training checkpoint и весов модели.
12. Вывод loss, LR обеих групп и времени эпохи.

## 5*. Сохранение и возобновление обучения (для большого кол-ва эпох)

Имена файлов автоматически зависят от имени архитектуры `timm`. Для
Tiny, например:

-   `model_last_convnext_tiny_in12k_ft_in1k_training_state.pt` ---
    checkpoint для продолжения обучения;
-   `model_convnext_tiny_in12k_ft_in1k.pt` --- чистый `state_dict`,
    предназначенный для ансамбля.

В training checkpoint записываются версия формата, номер эпохи, размер и
имя модели, веса модели, состояния оптимизатора, scheduler и GradScaler.
При `RESUME_TRAINING=True` и наличии checkpoint скрипт автоматически
продолжает со следующей эпохи. Проверяются совместимость архитектуры,
версия checkpoint и ожидаемое общее число шагов scheduler. Несовпадение
приводит к ошибке.

Чтобы начать обучение архитектуры с нуля, нужно удалить соответствующий
`model_last_..._training_state.pt` либо отключить `RESUME_TRAINING` в
конфигурации. При `KeyboardInterrupt` экспорт весов сохраняется; для
восстановления состояния оптимизатора и scheduler используется training
checkpoint. После окончания скрипт ещё раз сохраняет чистый `state_dict`
в `MODEL_EXPORT`.

**Важно:** для одной архитектуры имена checkpoint и экспорта
фиксированы. Повторный запуск `--model tiny` продолжит предыдущее
обучение и перезапишет тот же экспортный файл, если checkpoint
существует. Чтобы получить две независимые Tiny-модели, для второго
эксперимента нужно задать отдельные имена `LAST_CKPT` и `MODEL_EXPORT` в
копии скрипта (или сохранить первый экспорт, удалить checkpoint перед
новым запуском и сохранить второй экспорт под другим именем).

## 6. Ансамбль и инференс

### Запуск

``` bash
python comp2_ensemble.py \
  --weights tiny_1.pt tiny_2.pt small_1.pt \
  --models tiny tiny small \
  --output submission_ensemble.csv
```

`tiny_1.pt`, `tiny_2.pt`, `small_1.pt` --- примеры имён; укажите
фактические пути. Скрипт принимает любое количество моделей.

### Аргументы

| Аргумент | Обязательный | По умолчанию | Описание |
|---|:---:|---|---|
| `--weights` | Да | — | Список путей к файлам весов `.pt` |
| `--models` | Да | — | Архитектура каждой модели в том же порядке, что и `--weights`: `tiny`, `small` или `base` |
| `--batch-size` | Нет | `32` | Размер batch при инференсе |
| `--image-size` | Нет | `288` | Размер входного изображения |
| `--output` | Нет | `submission_ensemble.csv` | Имя итогового CSV-файла |
| `--blend-weights` | Нет | Равные коэффициенты | Неотрицательный коэффициент для каждой модели |

Число `--weights` и `--models` должно совпадать. Если передан
`--blend-weights`, число коэффициентов тоже должно совпадать; все
коэффициенты неотрицательны, сумма положительна. Скрипт нормализует их
до суммы 1.

### Загрузка весов

Для каждого файла создаётся соответствующая архитектура через
`timm.create_model(..., pretrained=False, num_classes=num_classes, drop_path_rate=0.1)`.
Скрипт поддерживает файл, содержащий непосредственно `state_dict`, либо
словарь с ключом `model` или `state_dict`. Префикс `module.` в начале
ключей удаляется. Загрузка выполняется с `strict=True`, поэтому
архитектура и число классов должны соответствовать сохранённым весам.
Модель переносится на устройство и переводится в `eval()`.

### Подготовка тестовых изображений

Порядок изображений задаётся строками `submission.csv`. Для каждого `id`
ищется файл в `data/test` по stem имени в нижнем регистре. Изображение
конвертируется в RGB, масштабируется до `288 × 288` (или
`--image-size`), преобразуется в Tensor и нормализуется теми же ImageNet
mean/std. DataLoader работает с `shuffle=False`, сохраняя порядок строк
шаблона.

### TTA: горизонтальное отражение

Для каждого batch модель получает исходные изображения и их
горизонтальное отражение (`torch.flip(..., dims=[3])`). Для обоих
проходов вычисляется softmax, после чего вероятности усредняются:

``` text
P_model = (softmax(model(images)) + softmax(model(flip(images)))) / 2
```



### Объединение моделей

Вероятности каждой модели объединяются взвешенной суммой:

``` text
P_ensemble = w1 * P_model_1 + w2 * P_model_2 + ... + wn * P_model_n
```

По умолчанию веса равны `1/N`. Например, задать коэффициенты
`0.4, 0.3, 0.3` можно так:

``` bash
python comp2_ensemble.py \
  --weights tiny_1.pt tiny_2.pt small_1.pt \
  --models tiny tiny small \
  --blend-weights 0.4 0.3 0.3 \
  --output submission_weighted.csv
```

После
объединения выбирается `argmax` по классам, предсказанные целые индексы
записываются в `label`. 

## 7. Сценарий для Kaggle

Для ансамбля из двух независимых Tiny и одной Small:

1.  Обучение Tiny1 и Tiny2: `python "comp2_train.py" --model tiny`.
2.  Сохранение экспортных весов как `tiny1.pt`, `tiny2.pt`.
3.  Обучение Small: `python "comp2_train.py" --model small` и
    сохранение экспорта как `small.pt`.
4.  Построение ансамбля:

``` bash
python comp2_ensemble.py --weights tiny1.pt tiny2.pt small.pt --models tiny tiny small --output submission_ensemble.csv
```

## 8. Результаты

Данный запуск создал файл, который дал 1.0000 F1 на public и 0.99529 на private, что в итоге - 2 картинки из 849 распознаны неверно.

Модель base в итоге не использовалась из-за тяжелого обучения и отсутствия улучшения результатов.