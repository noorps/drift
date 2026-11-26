# inference backends

one trained resnet18, 10000 cifar-10 test images, on a Tesla T4 (compute 7.5). each backend is diffed against eager fp32 at the same batch size, so the comparison isolates the backend rather than the batch.


## speed

| backend | batch | p50 ms | p99 ms | img/s | build s | peak mb | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager_fp32 | 1 | 2.651 | 3.807 | 377 | 0.0 | 184 | 0.9498 |
| eager_fp32 | 8 | 4.062 | 6.943 | 1970 | 0.0 | 188 | 0.9498 |
| eager_fp32 | 32 | 8.697 | 8.913 | 3680 | 0.0 | 219 | 0.9498 |
| eager_fp32 | 128 | 35.629 | 36.381 | 3593 | 0.0 | 475 | 0.9498 |
| eager_fp16 | 1 | 3.491 | 5.076 | 286 | 0.0 | 207 | 0.9499 |
| eager_fp16 | 8 | 3.449 | 4.827 | 2320 | 0.0 | 208 | 0.9499 |
| eager_fp16 | 32 | 4.424 | 5.754 | 7233 | 0.0 | 212 | 0.9500 |
| eager_fp16 | 128 | 14.607 | 14.840 | 8763 | 0.0 | 278 | 0.9500 |
| compile_fp32 | 1 | 2.453 | 2.629 | 408 | 9.6 | 229 | 0.9498 |
| compile_fp32 | 8 | 3.720 | 5.434 | 2150 | 1.4 | 226 | 0.9498 |
| compile_fp32 | 32 | 7.999 | 8.522 | 4000 | 0.0 | 338 | 0.9498 |
| compile_fp32 | 128 | 32.552 | 33.167 | 3932 | 0.0 | 861 | 0.9498 |
| compile_fp16 | 1 | 1.477 | 2.837 | 677 | 3.3 | 204 | 0.9500 |
| compile_fp16 | 8 | 2.394 | 2.895 | 3341 | 1.1 | 226 | 0.9498 |
| compile_fp16 | 32 | 4.077 | 4.674 | 7850 | 0.0 | 283 | 0.9498 |
| compile_fp16 | 128 | 13.027 | 13.664 | 9826 | 0.0 | 523 | 0.9498 |
| trt_fp32 | 1 | 1.171 | 1.380 | 854 | 9.8 | 177 | 0.9498 |
| trt_fp32 | 8 | 2.527 | 3.285 | 3166 | 8.1 | 177 | 0.9498 |
| trt_fp32 | 32 | 7.737 | 7.988 | 4136 | 8.5 | 177 | 0.9498 |
| trt_fp32 | 128 | 30.644 | 31.328 | 4177 | 14.0 | 177 | 0.9498 |
| trt_fp16 | 1 | 0.528 | 0.730 | 1894 | 16.7 | 223 | 0.9498 |
| trt_fp16 | 8 | 0.744 | 0.848 | 10752 | 15.0 | 248 | 0.9498 |
| trt_fp16 | 32 | 2.029 | 2.924 | 15775 | 15.6 | 271 | 0.9498 |
| trt_fp16 | 128 | 6.162 | 6.600 | 20771 | 18.3 | 272 | 0.9497 |

## numerics against eager fp32

headroom is the largest disagreement this backend produced as a fraction of the image's own top-two margin. below 1 means no prediction could change.

| backend | batch | abs max | rel max | identical | flips | headroom | acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager_fp16 | 1 | 0.0322 | 0.00378 | 0 | 1 | 2.04 | +0.0001 |
| eager_fp16 | 8 | 0.0322 | 0.00404 | 0 | 1 | 1.72 | +0.0001 |
| eager_fp16 | 32 | 0.0361 | 0.00392 | 0 | 2 | 2.03 | +0.0002 |
| eager_fp16 | 128 | 0.0362 | 0.00394 | 0 | 2 | 2.8 | +0.0002 |
| compile_fp32 | 1 | 9.54e-06 | 1.17e-06 | 0 | 0 | 0.00132 | +0.0000 |
| compile_fp32 | 8 | 4.77e-06 | 7.09e-07 | 0 | 0 | 0.000634 | +0.0000 |
| compile_fp32 | 32 | 3.34e-05 | 5.2e-06 | 0 | 0 | 0.00358 | +0.0000 |
| compile_fp32 | 128 | 3.24e-05 | 5.13e-06 | 0 | 0 | 0.00246 | +0.0000 |
| compile_fp16 | 1 | 0.0322 | 0.00376 | 0 | 2 | 2.45 | +0.0002 |
| compile_fp16 | 8 | 0.0322 | 0.00409 | 0 | 0 | 2.8 | +0.0000 |
| compile_fp16 | 32 | 0.0361 | 0.00386 | 0 | 0 | 2.79 | +0.0000 |
| compile_fp16 | 128 | 0.0362 | 0.00386 | 0 | 0 | 2.8 | +0.0000 |
| trt_fp32 | 1 | 1.19e-05 | 1.72e-06 | 0 | 0 | 0.00167 | +0.0000 |
| trt_fp32 | 8 | 9.06e-06 | 1.49e-06 | 0 | 0 | 0.00125 | +0.0000 |
| trt_fp32 | 32 | 3.67e-05 | 5.09e-06 | 0 | 0 | 0.00401 | +0.0000 |
| trt_fp32 | 128 | 3e-05 | 5.09e-06 | 0 | 0 | 0.00333 | +0.0000 |
| trt_fp16 | 1 | 0.0586 | 0.00711 | 0 | 4 | 17.8 | +0.0000 |
| trt_fp16 | 8 | 0.0617 | 0.00789 | 0 | 6 | 11.3 | +0.0000 |
| trt_fp16 | 32 | 0.0644 | 0.00692 | 0 | 5 | 18 | +0.0000 |
| trt_fp16 | 128 | 0.0569 | 0.00828 | 0 | 2 | 10.3 | -0.0001 |

## tolerance you would need

to compare a backend against eager fp32 in a test without false failures.

| backend | batch | atol | rtol |
| --- | ---: | ---: | ---: |
| eager_fp16 | 1 | 0.0322 | 0.00378 |
| eager_fp16 | 8 | 0.0322 | 0.00404 |
| eager_fp16 | 32 | 0.0361 | 0.00392 |
| eager_fp16 | 128 | 0.0362 | 0.00394 |
| compile_fp32 | 1 | 9.54e-06 | 1.17e-06 |
| compile_fp32 | 8 | 4.77e-06 | 7.09e-07 |
| compile_fp32 | 32 | 3.34e-05 | 5.2e-06 |
| compile_fp32 | 128 | 3.24e-05 | 5.13e-06 |
| compile_fp16 | 1 | 0.0322 | 0.00376 |
| compile_fp16 | 8 | 0.0322 | 0.00409 |
| compile_fp16 | 32 | 0.0361 | 0.00386 |
| compile_fp16 | 128 | 0.0362 | 0.00386 |
| trt_fp32 | 1 | 1.19e-05 | 1.72e-06 |
| trt_fp32 | 8 | 9.06e-06 | 1.49e-06 |
| trt_fp32 | 32 | 3.67e-05 | 5.09e-06 |
| trt_fp32 | 128 | 3e-05 | 5.09e-06 |
| trt_fp16 | 1 | 0.0586 | 0.00711 |
| trt_fp16 | 8 | 0.0617 | 0.00789 |
| trt_fp16 | 32 | 0.0644 | 0.00692 |
| trt_fp16 | 128 | 0.0569 | 0.00828 |

## about this machine

- no tf32 units on this device, so there is no tf32 row to measure
- no bf16 tensor core path on this device, so fp16 is the only reduced precision worth timing; a bf16 row would measure emulation

## baseline check

- eager fp32 at batch 1 reproduced stage 1's reference logits on 10000 of 10000 images bitwise
