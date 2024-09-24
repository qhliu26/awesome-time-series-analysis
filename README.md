<p align="center">
<img width="800" src="assets/fig/logo.png"/>
</p>

<!-- <h1 align="center">Awesome Time Series</h1> -->

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![[MIT License]](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/qhliu26/awesome-time-series)
[![Visits Badge](https://badges.pufler.dev/visits/qhliu26/awesome-time-series)](https://badges.pufler.dev/visits/qhliu26/awesome-time-series)


📖 A curated list of awesome time-series papers, benchmarks, datasets, tutorials.

If you find any missed resources or errors, please feel free to open an issue or make a PR.


## Main Recent Update


- [Sep. 22, 2024] Create the repo!

## Table of Contents
- [Introduction and Tutorials](#tutorial)
- [Time-series Analysis Tasks and Related Papers](#paper)
    * [Foundation-model Empowered Time-series Analysis](#foundation)
    * [Forecasting](#forecasting)
    * [Anomaly Detection](#tsad)
    * [Classification](#classification)
    * [Clustering](#clustering)
    * [Segmentation](#segmentation)
    * [Imputation](#imputation)

<h2 id="tutorial">Introduction and Tutorials</h2>

### Books and PhD Thesis
* Fast, Scalable, and Accurate Algorithms for Time-Series Analysis, *Phd Thesis* 2018. [\[Link\]](https://academiccommons.columbia.edu/doi/10.7916/D80K3S4B)
* Fast Algorithms for Mining Co-evolving Time Series, *Phd Thesis* 2010. [\[Link\]](https://lileicc.github.io/pubs/li2010fast.pdf)


### General Time Series Survey
* Foundation models for time series analysis: A tutorial and survey, in *KDD* 2024. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671451)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML* 2024. [\[Paper\]](https://arxiv.org/abs/2402.02713)
* Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.10196) [\[Website\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [\[Paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[Paper\]](https://arxiv.org/abs/2002.12478)


### Tutorial
* Time-Series Anomaly Detection: Overview and New Trends, in *VLDB* 2024. [\[Link\]](https://www.vldb.org/pvldb/vol17/p4229-liu.pdf) [\[Video\]](https://youtu.be/96869qimXAA?si=kww8SDL0HZ9CS4Y7)
* Out-of-Distribution Generalization in Time Series, in *AAAI* 2024. [\[Link\]](https://ood-timeseries.github.io/) [\[Slides\]](https://github.com/ood-timeseries/ood-timeseries.github.io/files/14325164/AAAI24_tutorial_OOD_in_time_series__slides_02182024.pdf)
* Robust Time Series Analysis and Applications: An Interdisciplinary Approach, in *ICDM* 2023. [\[Link\]](https://sites.google.com/view/timeseries-tutorial-icdm2023)

### Blogs
* TODO


<h2 id="paper">Time-series Analysis Tasks and Related Papers</h2>


<h3 id="foundation">⛰️ Foundation-model Empowered Time-series Analysis</h3>

* Agentic Retrieval-Augmented Generation for Time Series Analysis, in *KDD* 2024. [\[Paper\]](https://arxiv.org/pdf/2408.14484)
* Moment: A family of open time-series foundation models, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.03885)
* A decoder-only foundation model for time-series forecasting, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.10688)
* Chronos: Learning the language of time series, in *ArXiv* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.07815)
* Time-llm: Time series forecasting by reprogramming large language models, in *ICLR* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.01728)
* Lag-llama: Towards foundation models for time series forecasting, in *NeurIPS Workshop* 2023. [\[Paper\]](https://arxiv.org/pdf/2310.08278)
* One fits all: Power general time series analysis by pretrained lm, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/86c17de05579cde52025f9984e6e2ebb-Paper-Conference.pdf)
* Large Language Models Are Zero-Shot Time Series Forecasters, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf)


<h3 id="forecasting">☁️ Forecasting</h3>

<h4 id="forecasting-benchmark">Benchmark and Survey</h4>

* TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods, in *VLDB* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.20150)


<h4 id="forecasting-paper">Related Paper</h4>

* Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting, in *ICLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=vSVLM2j9eie)



<h3 id="tsad">⚙️ Anomaly Detection</h3>

<h4 id="tsad-benchmark">Benchmark and Survey</h4>

* An Experimental Evaluation of Anomaly Detection in Time Series, in *VLDB* 2023. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3632093.3632110?casa_token=k7Nl_Vgy4bQAAAAA:Xam85MABRXcLey5B9Ic_b7H4tzzpch_jz4jTWAi3D8PFnGFSOkZuPyCPnfVolmW_I7AXtmXbjDyth54)
* Timesead: Benchmarking deep multivariate time-series anomaly detection, in *TMLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=iMmsCI0JsS)
* TSB-UAD: an end-to-end benchmark suite for univariate time-series anomaly detection, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3529337.3529354?casa_token=JwNN0XtFhBwAAAAA:wa9QZshoY_Ib8As5sYDwcu7UY3IWuJ80FUE7eBhW3oazsiRBrGtRv1PmSeeFbhSx76o0RbZ41tiIaiU) [\[Website\]](https://github.com/TheDatumOrg/TSB-UAD)
* Anomaly detection in time series: a comprehensive evaluation, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3538598.3538602?casa_token=bNqpxgDjgGsAAAAA:X6NrQEHheNQPBG0W5AigQhInSlqoThMV4lgnZ6f_fRNg9Y5C7ECrdJCaPQVIb9ydZlSJO0SSkHFIy9o) [\[Website\]](https://timeeval.github.io/evaluation-paper)
* A review on outlier/anomaly detection in time series data, in *CSUR* 2021. [\[Paper\]](https://arxiv.org/abs/2002.04236)
* Anomaly detection for IoT time-series data: A survey, in *IEEE Internet of Things Journal* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8926446/?casa_token=nzFb6ihAXegAAAAA:TcN1je2Xp_9rIJyZ8O6Poq4mCmtjSoeoZZAJodgBKpGxzz84FFOUYlf028wPibQQAig5SRtZ0Q4)

<h4 id="tsad-paper">Related Paper</h4>



<h3 id="classification">🌴 Classification</h3>

<h4 id="classification-benchmark">Benchmark and Survey</h4>

* TODO

<h4 id="classification-paper">Related Paper</h4>

* TODO

<h3 id="clustering">🏖️ Clustering</h3>

<h4 id="clustering-benchmark">Benchmark and Survey</h4>

*  End-to-end deep representation learning for time series clustering: a comparative study, *Data Mining and Knowledge Discovery* 2023. [\[Paper\]](https://link.springer.com/article/10.1007/s10618-021-00796-y)

<h4 id="clustering-paper">Related Paper</h4>

* TODO

<h3 id="segmentation">🚪 Segmentation</h3>

<h4 id="segmentation-benchmark">Benchmark and Survey</h4>

* TODO

<h4 id="segmentation-paper">Related Paper</h4>

* ClaSP: parameter-free time series segmentation, in *Data Mining and Knowledge Discovery* 2023. [\[Paper\]](https://arxiv.org/pdf/2207.13987)



<h3 id="imputation">🧱 Imputation</h3>

<h4 id="imputation-benchmark">Benchmark and Survey</h4>

* TODO

<h4 id="imputation-paper">Related Paper</h4>

* TODO

* [Paper Title], in *[J/C]* [Year]. [\[Paper\]]()
