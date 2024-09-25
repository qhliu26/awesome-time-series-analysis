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
- [📚 Introduction and Tutorials](#tutorial)
- [📝 Time-series Analysis Tasks and Related Papers](#paper)
    * [Foundation-model Empowered Time-series Analysis](#foundation)
    * [General Time-series Analysis](#general)
    * [Forecasting](#forecasting)
    * [Anomaly Detection](#tsad)
    * [Classification](#classification)
    * [Clustering](#clustering)
    * [Segmentation](#segmentation)
    * [Imputation](#imputation)
- [📦 Awesome Time-series Analysis Toolkits](#toolkit)
- [🕶️ More Awesomeness](#related)


<h2 id="tutorial">📚 Introduction and Tutorials</h2>

### Books and PhD Thesis
* Fast, Scalable, and Accurate Algorithms for Time-Series Analysis, *Phd Thesis* 2018. [\[Link\]](https://academiccommons.columbia.edu/doi/10.7916/D80K3S4B)
* Fast Algorithms for Mining Co-evolving Time Series, *Phd Thesis* 2010. [\[Link\]](https://lileicc.github.io/pubs/li2010fast.pdf)


### General Time Series Survey
* Foundation models for time series analysis: A tutorial and survey, in *KDD* 2024. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671451)
* Position: What Can Large Language Models Tell Us about Time Series Analysis, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.02713)
* Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.10196) [\[Website\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [\[Paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[Paper\]](https://arxiv.org/abs/2002.12478)


### Tutorial
* An Introduction to Machine Learning from Time Series, in *ECML* 2024. [\[Link\]](https://aeon-tutorials.github.io/ECML-2024/code.html)
* Time-Series Anomaly Detection: Overview and New Trends, in *VLDB* 2024. [\[Link\]](https://www.vldb.org/pvldb/vol17/p4229-liu.pdf) [\[Video\]](https://youtu.be/96869qimXAA?si=kww8SDL0HZ9CS4Y7)
* Out-of-Distribution Generalization in Time Series, in *AAAI* 2024. [\[Link\]](https://ood-timeseries.github.io/) [\[Slides\]](https://github.com/ood-timeseries/ood-timeseries.github.io/files/14325164/AAAI24_tutorial_OOD_in_time_series__slides_02182024.pdf)
* Robust Time Series Analysis and Applications: An Interdisciplinary Approach, in *ICDM* 2023. [\[Link\]](https://sites.google.com/view/timeseries-tutorial-icdm2023)

### Blogs
* TODO


<h2 id="paper">📝 Time-series Analysis Tasks and Related Papers</h2>


<h3 id="foundation">⛰️ Foundation-model Empowered Time-series Analysis</h3>

* Agentic Retrieval-Augmented Generation for Time Series Analysis, in *KDD* 2024. [\[Paper\]](https://arxiv.org/pdf/2408.14484)
* Moment: A family of open time-series foundation models, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.03885) [\[Code\]](https://github.com/moment-timeseries-foundation-model/moment)
* A decoder-only foundation model for time-series forecasting, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.10688) [\[Code\]](https://github.com/google-research/timesfm)
* TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting, in *ICLR* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.04948) [\[Code\]](https://github.com/DC-research/TEMPO)
* Chronos: Learning the language of time series, in *arXiv* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.07815) [\[Code\]](https://github.com/amazon-science/chronos-forecasting)
* Time-llm: Time series forecasting by reprogramming large language models, in *ICLR* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.01728) [\[Code\]](https://github.com/KimMeen/Time-LLM)
* Lag-llama: Towards foundation models for time series forecasting, in *NeurIPS Workshop* 2023. [\[Paper\]](https://arxiv.org/pdf/2310.08278) [\[Code\]](https://github.com/time-series-foundation-models/lag-llama)
* One fits all: Power general time series analysis by pretrained lm, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/86c17de05579cde52025f9984e6e2ebb-Paper-Conference.pdf) [\[Code\]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
* Large Language Models Are Zero-Shot Time Series Forecasters, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf) [\[Code\]](https://github.com/ngruver/llmtime)


<h3 id="general">🧩 General Time-series Analysis</h3>

* FITS: Modeling Time Series with 10k Parameters, in *ICLR* 2024. [\[Paper\]](https://openreview.net/pdf?id=bWcnvZ3qMb) [\[Code\]](https://github.com/VEWOXIC/FITS)
* Timesnet: Temporal 2d-variation modeling for general time series analysis, in *ICLR* 2023. [\[Paper\]](https://arxiv.org/pdf/2210.02186v2/1000) [\[Code\]](https://github.com/thuml/TimesNet)
* Ts2vec: Towards universal representation of time series, in *AAAI* 2022. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/download/20881/version/19178/20640) [\[Code\]](https://github.com/yuezhihan/ts2vec)


<h3 id="forecasting">☁️ Forecasting</h3>

<h4 id="forecasting-benchmark">Benchmark and Survey</h4>

* TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods, in *VLDB* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.20150)


<h4 id="forecasting-paper">Related Paper</h4>

* Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting, in *ICLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=vSVLM2j9eie) [\[Code\]](https://github.com/Thinklab-SJTU/Crossformer)



<h3 id="tsad">⚙️ Anomaly Detection</h3>

<h4 id="tsad-benchmark">Benchmark and Survey</h4>

* An Experimental Evaluation of Anomaly Detection in Time Series, in *VLDB* 2023. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3632093.3632110?casa_token=k7Nl_Vgy4bQAAAAA:Xam85MABRXcLey5B9Ic_b7H4tzzpch_jz4jTWAi3D8PFnGFSOkZuPyCPnfVolmW_I7AXtmXbjDyth54)
* Timesead: Benchmarking deep multivariate time-series anomaly detection, in *TMLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=iMmsCI0JsS)
* TSB-UAD: an end-to-end benchmark suite for univariate time-series anomaly detection, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3529337.3529354?casa_token=JwNN0XtFhBwAAAAA:wa9QZshoY_Ib8As5sYDwcu7UY3IWuJ80FUE7eBhW3oazsiRBrGtRv1PmSeeFbhSx76o0RbZ41tiIaiU) [\[Website\]](https://github.com/TheDatumOrg/TSB-UAD)
* Anomaly detection in time series: a comprehensive evaluation, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3538598.3538602?casa_token=bNqpxgDjgGsAAAAA:X6NrQEHheNQPBG0W5AigQhInSlqoThMV4lgnZ6f_fRNg9Y5C7ECrdJCaPQVIb9ydZlSJO0SSkHFIy9o) [\[Website\]](https://timeeval.github.io/evaluation-paper)
* A review on outlier/anomaly detection in time series data, in *CSUR* 2021. [\[Paper\]](https://arxiv.org/abs/2002.04236)
* Anomaly detection for IoT time-series data: A survey, in *IEEE Internet of Things Journal* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8926446/?casa_token=nzFb6ihAXegAAAAA:TcN1je2Xp_9rIJyZ8O6Poq4mCmtjSoeoZZAJodgBKpGxzz84FFOUYlf028wPibQQAig5SRtZ0Q4)

<h4 id="tsad-paper">Related Paper</h4>

* TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data, in *VLDB* 2022. [\[Paper\]](http://vldb.org/pvldb/vol15/p1201-tuli.pdf) [\[Code\]](https://github.com/imperial-qore/TranAD)
* Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, in *ICLR* 2022. [\[Paper\]](https://openreview.net/pdf?id=LzQQ89U1qm_) [\[Code\]](https://github.com/thuml/Anomaly-Transformer)
* Usad: Unsupervised anomaly detection on multivariate time series, in *KDD* 2020. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392) [\[Code\]](https://github.com/manigalati/usad)
* Robust anomaly detection for multivariate time series through stochastic recurrent neural network, in *KDD* 2019. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330672) [\[Code\]](https://github.com/NetManAIOps/OmniAnomaly)
* Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications, in *WWW* 2018. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3178876.3185996) [\[Code\]](https://github.com/NetManAIOps/donut)


<h3 id="classification">🌴 Classification</h3>

<h4 id="classification-benchmark">Benchmark and Survey</h4>

*  Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[Paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)

<h4 id="classification-paper">Related Paper</h4>

* TODO

<h3 id="clustering">🏖️ Clustering</h3>

<h4 id="clustering-benchmark">Benchmark and Survey</h4>

*  End-to-end deep representation learning for time series clustering: a comparative study, in *Data Mining and Knowledge Discovery* 2023. [\[Paper\]](https://link.springer.com/article/10.1007/s10618-021-00796-y)

<h4 id="clustering-paper">Related Paper</h4>

* TODO

<h3 id="segmentation">🚪 Segmentation</h3>

<h4 id="segmentation-benchmark">Benchmark and Survey</h4>

* Unsupervised Time Series Segmentation: A Survey on Recent Advances, in *CMC* 2024. [\[Paper\]](https://cdn.techscience.cn/files/cmc/2024/TSP_CMC-80-2/TSP_CMC_54061/TSP_CMC_54061.pdf)
* An Evaluation of Change Point Detection Algorithms, in *arXiv* 2022. [\[Paper\]](https://arxiv.org/pdf/2003.06222) [\[Code\]](https://github.com/alan-turing-institute/TCPDBench)
* A survey of methods for time series change point detection, in *Knowledge and information systems* 2017. [\[Paper\]](https://www.researchgate.net/profile/Samaneh-Aminikhanghahi/publication/307947624_A_Survey_of_Methods_for_Time_Series_Change_Point_Detection/links/5806fcc008aeb85ac85f5cb6/A-Survey-of-Methods-for-Time-Series-Change-Point-Detection.pdf)

<h4 id="segmentation-paper">Related Paper</h4>

* ClaSP: parameter-free time series segmentation, in *Data Mining and Knowledge Discovery* 2023. [\[Paper\]](https://arxiv.org/pdf/2207.13987) [\[Code\]](https://github.com/ermshaua/claspy)
* Matrix profile VIII: domain agnostic online semantic segmentation at superhuman performance levels, in *ICDM* 2017. [\[Paper\]](http://www.cs.ucr.edu/~eamonn/Segmentation_ICDM.pdf) [\[Code\]](https://sites.google.com/site/onlinesemanticsegmentation/)
* Espresso: Entropy and shape aware time-series segmentation for processing heterogeneous sensor data, in *IMWUT* 2020. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3411832) [\[Code\]](https://github.com/cruiseresearchgroup/ESPRESSO/)



<h3 id="imputation">🧱 Imputation</h3>

<h4 id="imputation-benchmark">Benchmark and Survey</h4>

* TSI-Bench: Benchmarking Time Series Imputation, in *arXiv* 2024. [\[Paper\]](https://arxiv.org/pdf/2406.12747)

<h4 id="imputation-paper">Related Paper</h4>

* Brits: Bidirectional recurrent imputation for time series, in *NeurIPS* 2018. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf) [\[Code\]](https://github.com/caow13/BRITS)


* [Paper Title], in *[J/C]* [Year]. [\[Paper\]]()


<h2 id="toolkit">📦 Awesome Time-series Analysis Toolkits</h2>

* `aeon`: A toolkit for machine learning from time series. [\[Link\]](https://github.com/aeon-toolkit/aeon) ![Stars](https://img.shields.io/github/stars/aeon-toolkit/aeon)
* `sktime`: A unified framework for machine learning with time series. [\[Link\]](https://github.com/sktime/sktime) ![Stars](https://img.shields.io/github/stars/sktime/sktime)
* `Kats`: A toolkit to analyze time series data, a lightweight, easy-to-use, and generalizable framework to perform time series analysis. [\[Link\]](https://github.com/facebookresearch/Kats)  ![Stars](https://img.shields.io/github/stars/facebookresearch/Kats)
* `tsai`: State-of-the-art Deep Learning library for Time Series and Sequences. [\[Link\]](https://github.com/timeseriesAI/tsai)  ![Stars](https://img.shields.io/github/stars/timeseriesAI/tsai)
* `prophet`: Tool for producing high quality forecasts for time series data. [\[Link\]](https://github.com/facebook/prophet)  ![Stars](https://img.shields.io/github/stars/facebook/prophet)
* `darts`: A python library for user-friendly forecasting and anomaly detection on time series. [\[Link\]](https://github.com/unit8co/darts)  ![Stars](https://img.shields.io/github/stars/unit8co/darts)
* `gluonts`: Probabilistic time series modeling in Python. [\[Link\]](https://github.com/awslabs/gluonts)  ![Stars](https://img.shields.io/github/stars/awslabs/gluonts)
* `pyts`: A Python package for time series classification. [\[Link\]](https://github.com/johannfaouzi/pyts)  ![Stars](https://img.shields.io/github/stars/johannfaouzi/pyts)


<h2 id="related">🕶️ More Awesomeness</h2>

* [awesome-AI-for-time-series-papers](https://github.com/qingsongedu/awesome-AI-for-time-series-papers) ![Stars](https://img.shields.io/github/stars/qingsongedu/awesome-AI-for-time-series-papers)
* [Awesome-TimeSeries-SpatioTemporal-LM-LLM](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM) ![Stars](https://img.shields.io/github/stars/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* [awesome-time-series](https://github.com/lmmentel/awesome-time-series)  ![Stars](https://img.shields.io/github/stars/lmmentel/awesome-time-series)
* [awesome-time-series](https://github.com/cuge1995/awesome-time-series)  ![Stars](https://img.shields.io/github/stars/cuge1995/awesome-time-series)
* [TSFpaper](https://github.com/ddz16/TSFpaper)  ![Stars](https://img.shields.io/github/stars/ddz16/TSFpaper)
* [awesome-time-series-segmentation-papers](https://github.com/lzz19980125/awesome-time-series-segmentation-papers)  ![Stars](https://img.shields.io/github/stars/lzz19980125/awesome-time-series-segmentation-papers)
* [Awesome_Imputation](https://github.com/WenjieDu/Awesome_Imputation)  ![Stars](https://img.shields.io/github/stars/WenjieDu/Awesome_Imputation)
* [awesome-llm-time-series](https://github.com/xiyuanzh/awesome-llm-time-series)  ![Stars](https://img.shields.io/github/stars/xiyuanzh/awesome-llm-time-series)
* [Awesome-TimeSeries-LLM-FM](https://github.com/start2020/Awesome-TimeSeries-LLM-FM)  ![Stars](https://img.shields.io/github/stars/start2020/Awesome-TimeSeries-LLM-FM)

