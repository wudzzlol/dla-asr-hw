# HW1 ASR - DeepSpeech2 + CTC

Код и инструкция по запуску: https://github.com/wudzzlol/dla-asr-hw (README, demo.ipynb).

## Модель и обучение
- DeepSpeech2-like: 2 Conv2d + 4 BiGRU (hidden 512, bidirectional) + Linear, ~19M параметров
- Вход: log-mel спектрограмма, 128 мелов. Выход: 28 символов (blank + пробел + a–z + ')
- Лосс: CTC
- Данные: LibriSpeech train-clean-100 + train-clean-360 + train-other-500
- Аугментации (только на train):
  - на аудио: Gain $\pm$ 6 dB (p=0.5), цветной шум с SNR 15-30 dB (p=0.4), speed perturbation с коэффициентами 0.9 / 1.0 / 1.1
  - на спектрограмме: SpecAugment - frequency masking (F=20) и time masking (T=40)
- Оптимизация: AdamW, OneCycleLR (max_lr 1e-3, warmup 10% шагов, косинусное затухание), grad clipping 10, 65 эпох x 1000 шагов, batch 32, AMP, 2xT4, ~9.5 часов
- Лучший чекпоинт выбирался по WER (argmax) на валидации

![train loss](report_imgs/train_loss.jpg)

![val WER](report_imgs/wer_val.jpg)

Полные логи обучения: [ds2_full](https://www.comet.com/143xd/asr-hw/44kmyyzre8s421zyvdhoba7s6kmz7epm?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step) · [ds2_bpe](https://www.comet.com/143xd/asr-hw/canlf5m8sqd61dv6sm8h55ry41mg8p0m?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

Train loss резко падает в первые ~2k шагов, затем медленно снижается до ~0.25. Val WER падает с ~0.97 до ~0.4 за первые 10k шагов и дальше плавно снижается, выходя на плато ~0.17 после ~55k шагов. Переобучения нет - val WER не растет до конца обучения. Плато в конце совпадает с затуханием lr в OneCycle, так что при более длинном обучении, вероятно, можно было бы еще немного улучшить WER

## Как воспроизвести
Одно обучение с нуля, без дообучения: `python train.py -cn=ds2_train` (65 эпох). Инференс с LM - см. README (полная 3-gram, $\alpha$=0.8, $\beta$=3.0, beam 50)

## Декодирование
1. Argmax: CTC decode
2. Свой CTC prefix beam search (beam 10): для каждого префикса храню две вероятности: что он заканчивается на blank и что нет, чтобы правильно склеивать повторы. Токены с p < 1e-3 на кадре не рассматриваю, это сильно ускоряет поиск
3. Beam search + LM (shallow fusion через pyctcdecode, beam 50):  $\text{score} = \log P_{am} + \alpha \log P_{lm} + \beta \cdot N_{words}$. LM - LibriSpeech 3-gram (openslr), словарь librispeech-vocab

## Итоговые результаты (символьная модель)

| Сплит | Argmax WER / CER | Beam WER / CER | Beam + LM WER |
|---|---|---|---|
| test-clean | 17.02 / 5.17 | 16.44 / 5.00 | **9.50** |
| test-other | 36.74 / 14.17 | 36.03 / 13.85 | **24.26** |

Свой beam search стабильно лучше argmax на всех сплитах (~0.6-0.7 WER). Основной выигрыш дает LM: WER падает почти вдвое на test-clean

## Подбор $\alpha$, $\beta$ для LM
Подбор только на dev-other (500 случайных записей), тест не использовался

Урезанная LM (3-gram.pruned.1e-7), WER на dev-other:

| $\alpha$ \ $\beta$ | 1.0 | 2.0 | 3.0 |
|---|---|---|---|
| 0.3 | 25.98 | 26.77 | — |
| 0.5 | 23.69 | 24.28 | — |
| 0.6 | — | 23.75 | 24.16 |
| 0.7 | 23.34 | 23.16 | 23.52 |
| 0.8 | — | 23.04 | **22.97** |
| 0.9 | — | 23.81 | **22.97** |
| 1.0 | 26.75 | 24.99 | — |

Сравнение урезанной и полной 3-gram (другая подвыборка из 500 записей dev-other):

| LM | $\alpha$ | $\beta$ | WER |
|---|---|---|---|
| pruned | 0.8 | 3.0 | 21.56 |
| pruned | 0.8 | 3.5 | 21.81 |
| pruned | 0.8 | 4.0 | 22.14 |
| full | 0.7 | 2.0 | 21.11 |
| full | 0.8 | 3.0 | **21.01** |
| full | 0.8 | 4.0 | 21.31 |
| full | 0.9 | 3.0 | 21.75 |
| full | 0.9 | 4.0 | 21.18 |
| full | 1.0 | 4.0 | 22.13 |

Итог: полная 3-gram, $\alpha$=0.8, $\beta$=3.0. На test-other это дало 25.12 $\rightarrow$ 24.26 WER по сравнению с урезанной LM

## BPE (абляция)
Sentencepiece BPE, словарь 256 (+ blank), та же архитектура и те же гиперпараметры

| Сплит | Метод | Символы WER / CER | BPE WER / CER |
|---|---|---|---|
| test-clean | Argmax | 17.02 / 5.17 | 16.93 / 6.00 |
| test-clean | Beam | 16.44 / 5.00 | 16.54 / 5.86 |
| test-clean | Beam + LM (pruned) | 10.23 / 3.82 | 10.95 / 4.68 |
| test-other | Argmax | 36.74 / 14.17 | 36.25 / 16.00 |
| test-other | Beam | 36.03 / 13.85 | 35.83 / 15.81 |
| test-other | Beam + LM (pruned) | 25.12 / 12.03 | 25.16 / 13.59 |

Без LM BPE чуть лучше по WER, но заметно хуже по CER: ошибка в BPE сразу затрагивает кусок слова

С LM символьная модель лучше. Итого при такой маленькой модели и словаре 256 BPE не дал выигрыша. В качестве финальной модели использовалась символьная

В этой таблице обе модели декодировались с урезанной LM, поэтому числа с LM отличаются от итоговой таблицы, где используется полная 3-gram

## Что сработало / что нет
- Сработало: AMP + параллелизм на 2 GPU позволили обучить 65 эпох за адекватное количество времени; LM почти вдвое повысила качество; полная 3-gram вместо урезанной смогла еще немного повысить качество, преодолев порог в WER=25
- Не сработало: BPE не дал особого выигрыша по сравнению с символьной кодировкой; $\beta$ > 3 только ухудшает WER

## Сложности
Сборка kenlm на Python 3.13 (пришлось перегенерировать Cython обертку);
pyctcdecode откатывает numpy до 1.x

## Бонусы
- LM (shallow fusion, подбор $\alpha$/$\beta$ на dev-other)
- BPE + абляция против символьной модели
