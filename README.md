# Rethinking Discounting

The idea here is that discounting, in practice, is about reducing the impact of the uncertian future, where events distant in time are assumed to be more uncertian.

We instead discount on the uncertianty explicitly. That is, events that are a long way in the future but certian are not discounted.

Furthermore, this has the advantage that discounting can adapt over time (during training) and within specific parts of the environment. For example in game with two rooms, each room could have different discount factors.

## Requirements

```
pytorch
stable_baselines3
```

## Running the experiment

To run the code execute the following

``` python train_gamma.py ```

## Inspiration

Inspiration is taken from

* [Cluelessness](https://watermark.silverchair.com/aow018.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAtkwggLVBgkqhkiG9w0BBwagggLGMIICwgIBADCCArsGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM9KFlSC43ValkYYl6AgEQgIICjFnL3zqWZF0Z0_CM1CgOGgV_nPXKnNJdR03a8YdOQRmK2FyDvJQwmI4NrKxB3wZZDgB7KF6zEEo8sP7GG2soT8IvTIIuXhyqQr0kzE7xWgAqLgxD3KI0EdAHGbrh1jVgkn4Lv_5dYMgGnlxijN3aTKLy_jrpnvJfzYZGWFj6KG-5uaSsjXvPEV4qle4t-zfSoXV_XltaKEX_nPXbMvO-ROXWer_QS1nnSO4hGp3Ygpsn2bxkUnCEz4QGFevSH0PNuhLnEK94CRxf4ugO9wlbqzTrnhjsZhbWJur08TYS83m3DlUa5f79zEEVsK44RcgeFhZLIOrCPrfH5Hgysw9RZO_jt29snBcpCXhWsnvVz8yBmSFdS_duDVwrTYyL9qaKKxPx64YDECMGFsQLHNcTqtlDevAznw7evCNJnr6Am6m8ji-5fBbkLChYF_8jPLSsEWU-sFZkp4Y0hjjMKqscdcSDpxPvf01T35gF-jiflyEX0wDR83dAd3Nt3YRAW4Mf3BrvEhcvb3rRkXjx6VCUsvnhL90zl9HM3NQkNpjqp-WtH3mA1Kloy_Kzb-ai3q8OmsRDE0xQjRyRTGpxc5McbDHx2tEVkfk9An83RDtMBsXLFosbp7PYD3d9gTQ0LixeULUvs8OqLii7kjAgwhfHBxZqqkqCvMzODFmpxyTtjJTxrP8Rl_GsY-XDPNzdEJ_yTvj0srjcv4yRN92aFnSiFc5WXM-PUmgjJ3WUIZChlRJ5Ce4NxTC2ASiAk3knrOdHf40gpyYZQcLm2zZPqyOvWJ5ScCRRg299OF3-7l7xPMpZoSTs7WDjXq2SRk9oYttRVeVtGVaQS7IqDeGa8mpCZn0XbbKMv_KiLQE5Tbs) 
* [Bootstrap DQN](https://arxiv.org/abs/1602.04621)
* [Random Priors](https://arxiv.org/abs/1806.03335)
* [Random Network Distilation](https://arxiv.org/pdf/1810.12894.pdf)
* [Agent-57](https://deepmind.com/blog/article/Agent57-Outperforming-the-human-Atari-benchmark)
