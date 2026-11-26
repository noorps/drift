# drift results: t4

## environment

| | |
|---|---|
| gpu | Tesla T4 |
| driver | 580.82.07 |
| cuda | 13.0 |
| cudnn | 92800 |
| cpu | Intel(R) Xeon(R) CPU @ 2.00GHz (2 logical) |
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| python | 3.13.16 |
| os | Linux-6.6.159+-x86_64-with-glibc2.39 |

## resnet18 on cifar-10 (10,000 test images)

reference accuracy 94.98%. 0 of 10,000 came out with a different answer in at least one configuration, and 0 of those were in the lowest-margin 5%.

### what each knob does to the logits

| knob | compared against | max abs | median abs | max rel | median rel | bitwise identical | flips | time (base, other) |
|---|---|---|---|---|---|---|---|---|
| same config again, same process | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 29.0 s, 28.3 s |
| same config again, new process | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 29.0 s, 28.3 s |
| batch size 8 | gpu, batch 1, tf32 off | 8.11e-06 | 3.58e-07 | 1.49e-06 | 1.58e-07 | 0 / 10,000 | 0 | 29.0 s, 4.8 s |
| batch size 32 | gpu, batch 1, tf32 off | 3.86e-05 | 1.19e-06 | 4.67e-06 | 5.30e-07 | 0 / 10,000 | 0 | 29.0 s, 2.8 s |
| batch size 128 | gpu, batch 1, tf32 off | 2.91e-05 | 1.31e-06 | 5.05e-06 | 5.44e-07 | 0 / 10,000 | 0 | 29.0 s, 2.9 s |
| cpu instead of gpu | gpu, batch 1, tf32 off | 8.11e-06 | 4.17e-07 | 1.43e-06 | 1.78e-07 | 0 / 10,000 | 0 | 29.0 s, 214.6 s |
| tf32 on | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 29.0 s, 28.5 s |
| cudnn benchmark | gpu, batch 1, tf32 off | 3.81e-06 | 1.79e-07 | 3.09e-07 | 8.50e-08 | 0 / 10,000 | 0 | 29.0 s, 26.4 s |
| cudnn deterministic | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 29.0 s, 28.6 s |
| deterministic algorithms | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 29.0 s, 30.5 s |
| tf32 on, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 2.9 s, 2.8 s |
| cudnn benchmark, batch 128 | gpu, batch 128, tf32 off | 3.77e-05 | 1.67e-06 | 5.39e-06 | 6.83e-07 | 0 / 10,000 | 0 | 2.9 s, 2.3 s |
| cudnn deterministic, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 2.9 s, 2.8 s |
| deterministic algorithms, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 2.9 s, 2.9 s |
| batch 128 again, same process | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 2.9 s, 2.8 s |
| batch 128 again, new process | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 10,000 / 10,000 | 0 | 2.9 s, 2.9 s |

### flips by margin bucket

no pair flipped a prediction.

### divergence through depth

| pair | layers | first layer rel | last layer rel | ratio | first layer abs max | last layer abs max | ratio |
|---|---|---|---|---|---|---|---|
| gpu, batch 1, tf32 off vs gpu, batch 32, tf32 off | 69 | 9.61e-07 | 5.68e-07 | 0.6x | 3.96e-06 | 1.91e-05 | 4.8x |
| gpu, batch 1, tf32 off vs gpu, batch 128, tf32 off | 69 | 9.61e-07 | 5.56e-07 | 0.6x | 3.96e-06 | 1.96e-05 | 4.9x |
| gpu, batch 1, tf32 off vs cpu, batch 1 | 69 | 6.35e-08 | 1.82e-07 | 2.9x | 2.38e-07 | 4.77e-06 | 20.0x |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 on | 69 | bitwise identical at every layer | | | | | |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 off, cudnn autotune | 69 | 6.23e-07 | 8.33e-08 | 0.1x | 2.98e-07 | 1.91e-06 | 6.4x |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 off, deterministic mode | 69 | bitwise identical at every layer | | | | | |

## small transformer on ag news (7,600 test texts)

reference accuracy 90.67%. 0 of 7,600 came out with a different answer in at least one configuration, and 0 of those were in the lowest-margin 5%.

### what each knob does to the logits

| knob | compared against | max abs | median abs | max rel | median rel | bitwise identical | flips | time (base, other) |
|---|---|---|---|---|---|---|---|---|
| same config again, same process | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 12.9 s |
| same config again, new process | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 13.2 s |
| batch size 8 | gpu, batch 1, tf32 off | 1.91e-06 | 1.19e-07 | 1.36e-06 | 6.94e-08 | 261 / 7,600 | 0 | 12.5 s, 1.7 s |
| batch size 32 | gpu, batch 1, tf32 off | 1.91e-06 | 2.38e-07 | 1.47e-06 | 8.84e-08 | 45 / 7,600 | 0 | 12.5 s, 0.5 s |
| batch size 128 | gpu, batch 1, tf32 off | 1.91e-06 | 2.38e-07 | 1.47e-06 | 8.84e-08 | 45 / 7,600 | 0 | 12.5 s, 0.3 s |
| cpu instead of gpu | gpu, batch 1, tf32 off | 2.86e-06 | 2.38e-07 | 2.15e-06 | 1.00e-07 | 36 / 7,600 | 0 | 12.5 s, 15.0 s |
| tf32 on | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 12.7 s |
| cudnn benchmark | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 13.2 s |
| cudnn deterministic | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 12.4 s |
| deterministic algorithms | gpu, batch 1, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 12.5 s, 14.5 s |
| tf32 on, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.3 s |
| cudnn benchmark, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.3 s |
| cudnn deterministic, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.3 s |
| deterministic algorithms, batch 128 | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.3 s |
| batch 128 again, same process | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.2 s |
| batch 128 again, new process | gpu, batch 128, tf32 off | 0 | 0 | 0 | 0 | 7,600 / 7,600 | 0 | 0.3 s, 0.3 s |

### flips by margin bucket

no pair flipped a prediction.

### divergence through depth

| pair | layers | first layer rel | last layer rel | ratio | first layer abs max | last layer abs max | ratio |
|---|---|---|---|---|---|---|---|
| gpu, batch 1, tf32 off vs gpu, batch 32, tf32 off | 22 | 2.24e-07 | 8.86e-08 | 0.4x | 2.15e-06 | 1.43e-06 | 0.7x |
| gpu, batch 1, tf32 off vs gpu, batch 128, tf32 off | 22 | 2.24e-07 | 8.86e-08 | 0.4x | 2.15e-06 | 1.43e-06 | 0.7x |
| gpu, batch 1, tf32 off vs cpu, batch 1 | 22 | 7.42e-08 | 1.03e-07 | 1.4x | 9.54e-07 | 1.55e-06 | 1.6x |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 on | 22 | bitwise identical at every layer | | | | | |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 off, cudnn autotune | 22 | bitwise identical at every layer | | | | | |
| gpu, batch 1, tf32 off vs gpu, batch 1, tf32 off, deterministic mode | 22 | bitwise identical at every layer | | | | | |
