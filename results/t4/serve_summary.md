# inference backends

one trained resnet18, 10,000 cifar-10 test images, tesla t4 (compute 7.5), colab. each backend is
diffed against eager fp32 at the same batch size, so the comparison isolates the backend rather than
the batch. numbers come from three runs of `serve_stage2.py`; the tensorrt rows are from the run that
measured them alone, so the eager baseline is re-measured there rather than carried over.

## speed

| backend | batch | p50 ms | p99 ms | img/s | build s | peak mb | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager_fp32 | 1 | 2.579 | 2.888 | 388 | 0.0 | 184 | 0.9498 |
| eager_fp32 | 8 | 3.500 | 3.653 | 2286 | 0.0 | 188 | 0.9498 |
| eager_fp32 | 32 | 8.054 | 8.254 | 3973 | 0.0 | 219 | 0.9498 |
| eager_fp32 | 128 | 33.393 | 33.927 | 3833 | 0.0 | 475 | 0.9498 |
| eager_fp16 | 1 | 4.076 | 6.040 | 245 | 0.0 | 207 | 0.9499 |
| eager_fp16 | 8 | 3.232 | 4.888 | 2475 | 0.0 | 208 | 0.9499 |
| eager_fp16 | 32 | 4.150 | 4.718 | 7711 | 0.0 | 212 | 0.9500 |
| eager_fp16 | 128 | 13.816 | 13.924 | 9265 | 0.0 | 278 | 0.9500 |
| compile_fp32 | 1 | 2.368 | 2.590 | 422 | 21.1 | 231 | 0.9498 |
| compile_fp32 | 8 | 3.730 | 3.961 | 2145 | 18.8 | 226 | 0.9498 |
| compile_fp32 | 32 | 8.155 | 8.325 | 3924 | 0.0 | 338 | 0.9498 |
| compile_fp32 | 128 | 34.299 | 35.442 | 3732 | 0.0 | 861 | 0.9498 |
| compile_fp16 | 1 | 1.395 | 1.735 | 717 | 9.6 | 208 | 0.9500 |
| compile_fp16 | 8 | 1.958 | 2.151 | 4086 | 21.0 | 226 | 0.9498 |
| compile_fp16 | 32 | 4.059 | 4.463 | 7884 | 0.0 | 283 | 0.9498 |
| compile_fp16 | 128 | 13.020 | 13.310 | 9831 | 0.0 | 523 | 0.9498 |

tensorrt, measured in its own run against a freshly measured eager baseline:

| backend | batch | p50 ms | p99 ms | img/s | build s | peak mb | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager_fp32 | 1 | 2.505 | 4.327 | 399 | 0.0 | 196 | 0.9498 |
| eager_fp32 | 32 | 9.474 | 9.848 | 3378 | 0.0 | 240 | 0.9498 |
| eager_fp32 | 128 | 32.263 | 33.537 | 3967 | 0.0 | 454 | 0.9498 |
| trt_fp32 | 1 | 1.161 | 1.344 | 861 | 20.0 | 168 | 0.9498 |
| trt_fp32 | 32 | 8.297 | 8.408 | 3857 | 9.7 | 169 | 0.9498 |
| trt_fp32 | 128 | 38.165 | 39.146 | 3354 | 14.5 | 171 | 0.9498 |
| trt_fp16 | 1 | 1.223 | 1.309 | 818 | 3.7 | 168 | 0.9498 |
| trt_fp16 | 32 | 8.311 | 8.773 | 3850 | 4.1 | 169 | 0.9498 |
| trt_fp16 | 128 | 34.284 | 35.189 | 3734 | 4.2 | 171 | 0.9498 |

## numerics against eager fp32

headroom is the largest disagreement a backend produced as a fraction of the image's own top-two
margin. below 1 means no prediction could change. above 1 means at least one could, and did.

| backend | batch | abs max | rel max | flips | headroom | acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager_fp16 | 1 | 0.0322 | 0.00378 | 1 | 2.04 | +0.0001 |
| eager_fp16 | 8 | 0.0322 | 0.00404 | 1 | 1.72 | +0.0001 |
| eager_fp16 | 32 | 0.0361 | 0.00392 | 2 | 2.03 | +0.0002 |
| eager_fp16 | 128 | 0.0362 | 0.00394 | 2 | 2.8 | +0.0002 |
| compile_fp32 | 1 | 9.54e-06 | 1.17e-06 | 0 | 0.00132 | +0.0000 |
| compile_fp32 | 8 | 4.77e-06 | 7.09e-07 | 0 | 0.000634 | +0.0000 |
| compile_fp32 | 32 | 3.34e-05 | 5.2e-06 | 0 | 0.00358 | +0.0000 |
| compile_fp32 | 128 | 3.24e-05 | 5.13e-06 | 0 | 0.00246 | +0.0000 |
| compile_fp16 | 1 | 0.0322 | 0.00376 | 2 | 2.45 | +0.0002 |
| compile_fp16 | 8 | 0.0322 | 0.00409 | 0 | 2.8 | +0.0000 |
| compile_fp16 | 32 | 0.0361 | 0.00386 | 0 | 2.79 | +0.0000 |
| compile_fp16 | 128 | 0.0362 | 0.00386 | 0 | 2.8 | +0.0000 |
| trt_fp32 | 1 | 9.54e-06 | 1.59e-06 | 0 | 0.00125 | +0.0000 |
| trt_fp32 | 32 | 3.62e-05 | 5.43e-06 | 0 | 0.00329 | +0.0000 |
| trt_fp32 | 128 | 3.43e-05 | 5.63e-06 | 0 | 0.00336 | +0.0000 |
| trt_fp16 | 1 | 9.54e-06 | 1.59e-06 | 0 | 0.00125 | +0.0000 |
| trt_fp16 | 32 | 3.62e-05 | 5.43e-06 | 0 | 0.00329 | +0.0000 |
| trt_fp16 | 128 | 3.43e-05 | 5.63e-06 | 0 | 0.00336 | +0.0000 |

## what the numbers say

stage 1 swept batch size, device, tf32 and the cudnn autotuner and never found a single flipped
prediction. the margins were simply too wide for the divergence on offer. the closing section of that
stage guessed where flips would turn up, and put lower precision first on the list.

that guess holds. fp16 is the regime where it happens. the disagreement reaches two to three times the
smallest top-two margin in the set, and one or two images out of ten thousand change their answer. the
same sweep in fp32, compiled or through tensorrt, stays three orders of magnitude below the margin and
never flips anything.

the flips did not cost accuracy. fp16 came out 0.0001 to 0.0002 ahead, so the predictions that moved
were near-ties that happened to land on the right side. a flip is a stability fact, not an accuracy fact.

no backend wins everywhere, which is the part worth carrying away:

- at batch 1 tensorrt is fastest, 861 img/s against eager's 399, and it is numerically tight enough to
  swap in without changing a single prediction
- at batch 128 tensorrt is the slowest option measured, 3,354 img/s against eager's 3,967, while
  torch.compile with fp16 reaches 9,831
- torch.compile in fp32 is the free one: it never flipped a prediction at any batch size and stays within
  4.5e-05 of eager, so it costs nothing but compile time

tensorrt also holds memory flat as the batch grows, 168 to 171 mb across the whole range, against eager's
196 to 454 mb. at batch 128 that is 2.7x less memory for 15% less throughput.

eager fp16 at batch 1 is slower than eager fp32, 4.076 ms against 2.579 ms. autocast has to pay for the
casts and there is not enough tensor core work at batch 1 to earn it back. only once inductor fuses the
graph does fp16 win at that size.

## tolerance you would need

to diff a backend against eager fp32 in a test without tripping on precision noise.

| backend | atol | rtol |
| --- | ---: | ---: |
| compile_fp32 | 4.5e-05 | 6.3e-06 |
| trt_fp32 | 3.7e-05 | 5.7e-06 |
| trt_fp16 | 3.7e-05 | 5.7e-06 |
| eager_fp16 | 0.037 | 0.0041 |
| compile_fp16 | 0.037 | 0.0041 |

three orders of magnitude between the fp32 paths and the fp16 ones. a single tolerance covering both
would be loose enough to hide a real regression.

## where it falls short

- one gpu, one card, one architecture. the t4 is turing, so there are no tf32 units to measure and no
  bf16 tensor core path worth timing; a bf16 row would be measuring emulation
- gpu clocks could not be locked on colab. the same eager fp32 row came out 8.054 ms in one run and
  9.474 ms in another, about 18% apart, so small gaps between backends in these tables mean nothing
- **the tensorrt fp16 rows are not confirmed fp16.** trt_fp16 and trt_fp32 agree to the last bit at every
  batch size, which means the fp16 kernels were almost certainly never selected. the harness flags this
  itself rather than reporting an fp16 result that is really an fp32 one
- int8 is untested, and turing does have int8 tensor cores. that is the regime where flips should become
  common rather than rare, and it is the obvious next run
- no serving layer. this measures inference libraries, not a server, so there is no queue delay, no
  dynamic batching and no concurrency
- one model family for the headline. the ag news transformer has not been through this pipeline yet
- stage 1's reference logits were not on disk under this label, so the baseline self-check was skipped
