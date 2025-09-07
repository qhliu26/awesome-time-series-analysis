<p align="center">
<img width="800" src="assets/fig/logo.png"/>
</p>

<!-- <h1 align="center">Awesome Time Series</h1> -->

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![[MIT License]](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 

📖 A curated list of awesome time-series papers, benchmarks, datasets, tutorials.

If you find any missed resources or errors, please feel free to open an issue or make a PR.


## Main Recent Update

- [Sep. 7, 2025] Add papers on AI in Finance
- [Aug. 15, 2025] Add papers from ICML 2025
- [Mar. 5, 2025] Add papers from ICLR 2025
- [Dec. 19, 2024] Add papers from NeurIPS 2024
- [Sep. 22, 2024] Create the repo!

## Table of Contents
- [📚 Introduction and Tutorials](#tutorial)
- [📝 Time-series Analysis Tasks and Related Papers](#paper)
    * [Time-series Analysis In General](#general)
    * [Forecasting](#forecasting)
    * [Anomaly Detection](#tsad)
    * [Classification](#classification)
    * [Clustering](#clustering)
    * [Segmentation](#segmentation)
    * [Imputation](#imputation)
- [📈 AI in Finance](#fin)
- [📦 Awesome Time-series Analysis Toolkits](#toolkit)
- [🕶️ More Awesomeness](#related)


<h2 id="tutorial">📚 Introduction and Tutorials</h2>

### Books and PhD Thesis
* Forecasting: Principles and Practice. [\[Link\]](https://otexts.com/fpp2/)
* Fast, Scalable, and Accurate Algorithms for Time-Series Analysis, *Phd Thesis* 2018. [\[Link\]](https://academiccommons.columbia.edu/doi/10.7916/D80K3S4B)
* Fast Algorithms for Mining Co-evolving Time Series, *Phd Thesis* 2010. [\[Link\]](https://lileicc.github.io/pubs/li2010fast.pdf)


### Workshops and Tutorials
* Time Series in the Age of Large Models, in *NeurIPS* 2024. [\[Link\]](https://neurips-time-series-workshop.github.io/)
* An Introduction to Machine Learning from Time Series, in *ECML* 2024. [\[Link\]](https://aeon-tutorials.github.io/ECML-2024/code.html)
* Time-Series Anomaly Detection: Overview and New Trends, in *VLDB* 2024. [\[Link\]](https://www.vldb.org/pvldb/vol17/p4229-liu.pdf) [\[Video\]](https://youtu.be/96869qimXAA?si=kww8SDL0HZ9CS4Y7)
* AI for Time Series Analysis, in *AAAI* 2024. [\[Link\]](https://ai4ts.github.io/aaai2024)
* Out-of-Distribution Generalization in Time Series, in *AAAI* 2024. [\[Link\]](https://ood-timeseries.github.io/) [\[Slides\]](https://github.com/ood-timeseries/ood-timeseries.github.io/files/14325164/AAAI24_tutorial_OOD_in_time_series__slides_02182024.pdf)
* Robust Time Series Analysis and Applications: An Interdisciplinary Approach, in *ICDM* 2023. [\[Link\]](https://sites.google.com/view/timeseries-tutorial-icdm2023)


<h2 id="paper">📝 Time-series Papers</h2>

<h3 id="general">🧩 Time-series Analysis In General</h3>

<h4 id="general-benchmark">Benchmark and Survey</h4>

* Foundation models for time series analysis: A tutorial and survey, in *KDD* 2024. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3637528.3671451)
* Position: What Can Large Language Models Tell Us about Time Series Analysis, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.02713)
* Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook, in *arXiv* 2023. [\[Paper\]](https://arxiv.org/abs/2310.10196) [\[Website\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [\[Paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[Paper\]](https://arxiv.org/abs/2002.12478)

<h4 id="general-paper">Related Paper</h4>

* Sundial: A Family of Highly Capable Time Series Foundation Models, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2502.00816) [\[Code\]](https://github.com/thuml/Sundial)
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=1CLzLXSFNn) [\[Code\]](https://github.com/kwuking/TimeMixer)
* Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=e1wDDFmlVu) [\[Code\]](https://github.com/Time-MoE/Time-MoE)
* UniTS: A unified multi-task time series model, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=nBOdYBptWW) [\[Code\]](https://github.com/mims-harvard/UniTS)
* Agentic Retrieval-Augmented Generation for Time Series Analysis, in *KDD* 2024. [\[Paper\]](https://arxiv.org/pdf/2408.14484)
* Moment: A family of open time-series foundation models, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.03885) [\[Code\]](https://github.com/moment-timeseries-foundation-model/moment)
* Timer: Generative Pre-trained Transformers Are Large Time Series Models, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.02368) [\[Code\]](https://github.com/thuml/Large-Time-Series-Model)
* Unified Training of Universal Time Series Forecasting Transformers, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.02592) [\[Code\]](https://github.com/SalesforceAIResearch/uni2ts)
* A decoder-only foundation model for time-series forecasting, in *ICML* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.10688) [\[Code\]](https://github.com/google-research/timesfm)
* TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting, in *ICLR* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.04948) [\[Code\]](https://github.com/DC-research/TEMPO)
* Chronos: Learning the language of time series, in *arXiv* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.07815) [\[Code\]](https://github.com/amazon-science/chronos-forecasting)
* Time-llm: Time series forecasting by reprogramming large language models, in *ICLR* 2024. [\[Paper\]](https://arxiv.org/pdf/2310.01728) [\[Code\]](https://github.com/KimMeen/Time-LLM)
* FITS: Modeling Time Series with 10k Parameters, in *ICLR* 2024. [\[Paper\]](https://openreview.net/pdf?id=bWcnvZ3qMb) [\[Code\]](https://github.com/VEWOXIC/FITS)
* Lag-llama: Towards foundation models for time series forecasting, in *NeurIPS Workshop* 2023. [\[Paper\]](https://arxiv.org/pdf/2310.08278) [\[Code\]](https://github.com/time-series-foundation-models/lag-llama)
* One fits all: Power general time series analysis by pretrained lm, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/86c17de05579cde52025f9984e6e2ebb-Paper-Conference.pdf) [\[Code\]](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
* Large Language Models Are Zero-Shot Time Series Forecasters, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/3eb7ca52e8207697361b2c0fb3926511-Paper-Conference.pdf) [\[Code\]](https://github.com/ngruver/llmtime)
* A Shapelet-based Framework for Unsupervised Multivariate Time Series Representation Learning, in *VLDB* 2023. [\[Paper\]](https://www.vldb.org/pvldb/vol17/p386-wang.pdf) [\[Code\]](https://github.com/real2fish/CSL)
* Timesnet: Temporal 2d-variation modeling for general time series analysis, in *ICLR* 2023. [\[Paper\]](https://arxiv.org/pdf/2210.02186v2/1000) [\[Code\]](https://github.com/thuml/TimesNet)
* Ts2vec: Towards universal representation of time series, in *AAAI* 2022. [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/download/20881/version/19178/20640) [\[Code\]](https://github.com/yuezhihan/ts2vec)


<h3 id="forecasting">☁️ Forecasting</h3>

<h4 id="forecasting-benchmark">Benchmark and Survey</h4>

* TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods, in *VLDB* 2024. [\[Paper\]](https://arxiv.org/pdf/2403.20150)


<h4 id="forecasting-paper">Related Paper</h4>

* Patch-wise Structural Loss for Time Series Forecasting, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2503.00877) [\[Code\]](https://github.com/Dilfiraa/PS_Loss)
* TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2410.04442) [\[Code\]](https://github.com/Hank0626/TimeBridge)
* IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2505.22815) [\[Code\]](https://github.com/WHU-HZY/VIMTS)
* VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecastersg, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2408.17253) [\[Code\]](https://github.com/Keytoyze/VisionTS)
* Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2502.04395) [\[Code\]](https://github.com/CityMind-Lab/ICML25-TimeVLM)
* AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2502.10235) [\[Code\]](https://github.com/abenechehab/AdaPTS)
* Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2412.05244)
* TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2410.05711) [\[Code\]](https://github.com/Melmaphother/TimeDART)
* Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=uxVBbSlKQ4)
* CoMRes: Semi-Supervised Time Series Forecasting Utilizing Consensus Promotion of Multi-Resolution, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=bRa4JLPzii) [\[Code\]](https://github.com/yjucho1/CoMRes)
* Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2401.03955) [\[Code\]](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer)
* TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2402.19072) [\[Code\]](https://github.com/thuml/TimeXer)
* From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2409.17515) [\[Code\]](https://github.com/ameliawong1996/From_News_to_Forecast)
* CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=clBiQUgj4w) [\[Code\]](https://github.com/ACAT-SCUT/CycleNet)
* Are Language Models Actually Useful for Time Series Forecasting, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/abs/2406.16964) [\[Code\]](https://github.com/BennyTMT/LLMsForTimeSeries)
* Are Self-Attentions Effective for Time Series Forecasting, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2405.16877) [\[Code\]](https://github.com/dongbeank/CATS)
* Frequency Adaptive Normalization For Non-stationary Time Series Forecasting, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=T0axIflVDD) [\[Code\]](http://github.com/wayne155/FAN)
* DDN: Dual-domain Dynamic Normalization for Non-stationary Time Series Forecasting, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=RVZfra6sZo) [\[Code\]](https://github.com/Hank0626/DDN)
* Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting, in *ICLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=vSVLM2j9eie) [\[Code\]](https://github.com/Thinklab-SJTU/Crossformer)
* Reversible instance normalization for accurate time-series forecasting against distribution shift, in *ICLR* 2022. [\[Paper\]](https://openreview.net/pdf?id=cGDAkQo1C0p) [\[Code\]](https://github.com/ts-kim/RevIN)



<h3 id="tsad">⚙️ Anomaly Detection</h3>

<h4 id="tsad-benchmark">Benchmark and Survey</h4>

* The Elephant in the Room: Towards A Reliable Time-series Anomaly Detection Benchmark, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=R6kJtWsTGy) [\[Website\]](https://thedatumorg.github.io/TSB-AD/)
* Deep learning for time series anomaly detection: A survey, in *CSUR* 2024. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3691338)
* An Experimental Evaluation of Anomaly Detection in Time Series, in *VLDB* 2023. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3632093.3632110?casa_token=k7Nl_Vgy4bQAAAAA:Xam85MABRXcLey5B9Ic_b7H4tzzpch_jz4jTWAi3D8PFnGFSOkZuPyCPnfVolmW_I7AXtmXbjDyth54)
* Timesead: Benchmarking deep multivariate time-series anomaly detection, in *TMLR* 2023. [\[Paper\]](https://openreview.net/pdf?id=iMmsCI0JsS)
* TSB-UAD: an end-to-end benchmark suite for univariate time-series anomaly detection, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3529337.3529354?casa_token=JwNN0XtFhBwAAAAA:wa9QZshoY_Ib8As5sYDwcu7UY3IWuJ80FUE7eBhW3oazsiRBrGtRv1PmSeeFbhSx76o0RbZ41tiIaiU) [\[Website\]](https://github.com/TheDatumOrg/TSB-UAD)
* Anomaly detection in time series: a comprehensive evaluation, in *VLDB* 2022. [\[Paper\]](https://dl.acm.org/doi/pdf/10.14778/3538598.3538602?casa_token=bNqpxgDjgGsAAAAA:X6NrQEHheNQPBG0W5AigQhInSlqoThMV4lgnZ6f_fRNg9Y5C7ECrdJCaPQVIb9ydZlSJO0SSkHFIy9o) [\[Website\]](https://timeeval.github.io/evaluation-paper)
* A review on outlier/anomaly detection in time series data, in *CSUR* 2021. [\[Paper\]](https://arxiv.org/abs/2002.04236)
* Anomaly detection for IoT time-series data: A survey, in *IEEE Internet of Things Journal* 2019. [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/8926446/?casa_token=nzFb6ihAXegAAAAA:TcN1je2Xp_9rIJyZ8O6Poq4mCmtjSoeoZZAJodgBKpGxzz84FFOUYlf028wPibQQAig5SRtZ0Q4)


<h4 id="tsad-paper">Related Paper</h4>


* KAN-AD: Time Series Anomaly Detection with Kolmogorov–Arnold Networks, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2411.00278) [\[Code\]](https://github.com/CSTCloudOps/KAN-AD)
* Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2506.03964) [\[Code\]](https://github.com/kimanki/CAROTS)
* Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=aKcd7ImG5e) [\[Code\]](https://github.com/decisionintelligence/DADA)
* Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=eWocmTQn7H)
* CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=m08aK3xxdJ) [\[Code\]](https://github.com/decisionintelligence/CATCH)
* Can LLMs Understand Time Series Anomalies, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=k38Th3x4d9) [\[Code\]](https://github.com/rose-stl-lab/anomllm)
* Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=k38Th3x4d9) [\[Code\]](https://github.com/hanxiao0607/AERCA)
* SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=gmf5Aj01Hz) [\[Code\]](https://github.com/daidahao/SARAD)
* Memto: Memory-guided transformer for multivariate time series anomaly detection, in *NeurIPS* 2023. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b4c898eb1fb556b8d871fbe9ead92256-Paper-Conference.pdf) [\[Code\]](https://github.com/gunny97/MEMTO)
* Dcdetector: Dual attention contrastive representation learning for time series anomaly detection, in *KDD* 2023. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599295?casa_token=6CwHyakbA1kAAAAA:dEw8Iwjk8OdIpA-uBRVqRjmS4_HVazkCN4bh7ugRHeQCYUHz3lSKX8agKCwzvwRnWilnOZtqocI4vTY) [\[Code\]](https://github.com/DAMO-DI-ML/KDD2023-DCdetector)
* TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data, in *VLDB* 2022. [\[Paper\]](http://vldb.org/pvldb/vol15/p1201-tuli.pdf) [\[Code\]](https://github.com/imperial-qore/TranAD)
* Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, in *ICLR* 2022. [\[Paper\]](https://openreview.net/pdf?id=LzQQ89U1qm_) [\[Code\]](https://github.com/thuml/Anomaly-Transformer)
* TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis, in *ICLR* 2022. [\[Paper\]](https://arxiv.org/pdf/2210.09693) [\[Code\]](https://github.com/DAMO-DI-ML/CIKM22-TFAD)
* Usad: Unsupervised anomaly detection on multivariate time series, in *KDD* 2020. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392) [\[Code\]](https://github.com/manigalati/usad)
* Robust anomaly detection for multivariate time series through stochastic recurrent neural network, in *KDD* 2019. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3292500.3330672) [\[Code\]](https://github.com/NetManAIOps/OmniAnomaly)
* Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications, in *WWW* 2018. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3178876.3185996) [\[Code\]](https://github.com/NetManAIOps/donut)


<h3 id="classification">🌴 Classification</h3>

<h4 id="classification-benchmark">Benchmark and Survey</h4>

*  Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[Paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)

<h4 id="classification-paper">Related Paper</h4>

* Spectral-Aware Reservoir Computing for Fast and Accurate Time Series Classification, in *ICML* 2025. [\[Paper\]](https://openreview.net/pdf?id=DmPW0pO3F3) [\[Code\]](https://github.com/ZOF-pt/SARC)
* Learning Soft Sparse Shapes for Efficient Time-Series Classification, in *ICML* 2025. [\[Paper\]](https://arxiv.org/pdf/2505.06892) [\[Code\]](https://github.com/qianlima-lab/SoftShape)
* UniMTS: Unified Pre-training for Motion Time Series, in *NeurIPS* 2024. [\[Paper\]](https://openreview.net/pdf?id=DpByqSbdhI) [\[Code\]](https://github.com/xiyuanzh/UniMTS)
* Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2411.01006) [\[Code\]](https://github.com/YunshiWen/VQShape)
* Con4m: Context-aware Consistency Learning Framework for Segmented Time Series Classification, in *NeurIPS* 2024. [\[Paper\]](https://arxiv.org/pdf/2408.00041) [\[Code\]](https://github.com/MrNobodyCali/Con4m)

<h3 id="clustering">🏖️ Clustering</h3>

<h4 id="clustering-benchmark">Benchmark and Survey</h4>

*  End-to-end deep representation learning for time series clustering: a comparative study, in *Data Mining and Knowledge Discovery* 2023. [\[Paper\]](https://link.springer.com/article/10.1007/s10618-021-00796-y)
*  Clustering of time series data—a survey, in *Pattern Recognition* 2005. [\[Paper\]](https://www.sciencedirect.com/science/article/pii/S0031320305001305?casa_token=55dVI7qJgOQAAAAA:C8u2o_lgFkiIXwvheIhORs-BjMLIqTBZWsq18-VX_jVkRdo5-w8LuQPYwizemvQufNBZb4GvvbS1)


<h4 id="clustering-paper">Related Paper</h4>

* k-shape: Efficient and accurate clustering of time series, in *SIGMOD* 2015. [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/2723372.2737793?casa_token=xULrZ0SDZYMAAAAA:krad1sEMVW4Sa5GO6qgDI6k5XC2pyp9cikSp95prI2uYTqZWnybYdmignJ-ObydUAPKOin_zjbONuLQ) [\[Code\]](https://github.com/TheDatumOrg/kshape-python)

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

* LSCD: Lomb-Scargle Conditioned Diffusion for Time series Imputation, in *ICML* 2025. [\[Paper\]](https://www.arxiv.org/pdf/2506.17039)
* Optimal Transport for Time Series Imputation, in *ICLR* 2025. [\[Paper\]](https://openreview.net/pdf?id=xPTzjpIQNp) [\[Code\]](https://github.com/FMLYD/PSW-I)
* Brits: Bidirectional recurrent imputation for time series, in *NeurIPS* 2018. [\[Paper\]](https://proceedings.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf) [\[Code\]](https://github.com/caow13/BRITS)

<h2 id="fin">📈 AI in Finance</h2>

* The LDBC Financial Benchmark: Transaction Workload, in *VLDB* 2025. [\[Paper\]](https://www.vldb.org/pvldb/vol18/p3007-qi.pdf) [\[Code\]](https://github.com/ldbc/ldbc_finbench_transaction_impls)
* FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering, in *Arxiv* 2025. [\[Paper\]](https://arxiv.org/pdf/2508.14052) 
* TradingAgents: Multi-Agents LLM Financial Trading Framework, in *Arxiv* 2024. [\[Paper\]](https://arxiv.org/pdf/2412.20138) [\[Code\]](https://github.com/TauricResearch/TradingAgents)


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

