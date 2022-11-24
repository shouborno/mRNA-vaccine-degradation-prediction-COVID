# COVID-19 mRNA Vaccine Degradation Prediction using Regularized LSTM Model
Official repository for the paper "COVID-19 mRNA Vaccine Degradation Prediction using Regularized LSTM Model." Please run `pip install -r requirements.txt`. The dataset, if missing, would be automatically downloaded when `python train.py` is run after following https://github.com/Kaggle/kaggle-api#api-credentials to setup `~/.kaggle/kaggle.json`. 

## [Paper][paper]
[paper]: https://ieeexplore.ieee.org/document/9398044

<br>
<i>Dataset:</i>

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)][data]

[data]: https://www.kaggle.com/competitions/stanford-covid-vaccine


Abstract: <i>Due to the advantages of mRNA vaccines such as potency, safety, and production feasibility, recent researches in vaccinology has seen strong focus in mRNA vaccines. As leading researches involving COVID-19 mRNA vaccine candidates are being carried out, the challenge of overcoming the stability tradeoff of mRNA vaccines stand between the production and effective mass distribution stages. With the help of the OpenVaccine RNA database with degradation rate measurements provided by Stanford researchers, we developed an artificial recurrent neural network model to help bioinformatics researchers identify whether and where mRNAs might be unstable and prone to degrade under certain incubation measures. For this purpose we’ve prepared a regularized LSTM model which minimizes mean columnwise root mean squared error for several degradation rates. We’ve found that recurrent algorithms perform better than tree-based algorithms.</i>

## Citation
```
@INPROCEEDINGS{imran2020covid,
  author={Asif Imran, Sheikh and Tariqul Islam, Md. and Shahnaz, Celia and Tafhimul Islam, Md. and Tawhid Imam, Omar and Haque, Moinul},
  booktitle={2020 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE)}, 
  title={COVID-19 mRNA Vaccine Degradation Prediction using Regularized LSTM Model}, 
  year={2020},
  pages={328-331},
  doi={10.1109/WIECON-ECE52138.2020.9398044}}
```
