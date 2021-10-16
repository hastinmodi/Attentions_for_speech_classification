This repo contains the code for the paper 'Attentions for short duration speech classification'. 

There are 2 folders named Emotion_recognition and Infant_cry which contain the codes for the respective tasks. The files named Transformer  and ResNeSt contain the code for the Transformer and the ResNeSt architecture as described in the paper and the files named test_Transformer and test_ResNeSt contain the code for finding the performance of the models by removing 15 and 20% of the frames.
The infant cry database (Baby Chillanto) contains audio clips of a duration of 1 second while the emotion recognition database (CREMA-D - https://github.com/CheyneyComputerScience/CREMA-D) contains audio clips of upto 5 seconds because of which we have obtained results using 1, 3 and, 5 seconds of durations (for high intensity emotion level only).

The code has been written using Python 3.6.5 and TensorFlow 2.2.0

The results obtained are as follows:

## Infant cry

#### Performance (mean + std.) of Transformers

|% of frames removed|AUC|Recall|F1|Accuracy|
|:----:|:---------:|:-:|:----:|:------:|
|0%|0.9898|0.9706|0.9296|96.4%|
|15%|0.9896 (+/- 0.001)|0.92944 (+/- 0.012)|0.90414 (+/- 0.0084)|95.2% (+/- 0.4%)|
|20%|0.9892 (+/- 0.001)|0.98824 (+/- 0.007)|0.91432 (+/- 0.0075)|95.4% (+/- 0.4%)|

#### Performance (mean + std.) of ResNeSt

|% of frames removed|AUC|Recall|F1|Accuracy|
|:----:|:---------:|:-:|:----:|:------:|
|0%|0.9821|0.8971|0.9037|95.32%|
|15%|0.981 (+/- 0.002)|0.8676 (+/- 0.01)|0.8846 (+/- 0.012)|94.5% (+/- 0.65%)|
|20%|0.987 (+/- 0.001)|0.8736 (+/- 0.04)|0.885 (+/- 0.027)|94.5% (+/- 1.2%)|

## Emotion recognition 

#### Performance (mean + std.) of Transformers

|Duration|% of frames removed|AUC|F1|Accuracy|
|:----:|:---------:|:-:|:----:|:------:|
|1 sec|0%|0.8247|0.4654|52.2%|
| |15%|0.7982 (+/- 0.011)|0.4174 (+/- 0.008)|48.89% (+/- 0.78%)
| |20%|0.7948 (+/- 0.012)|0.2319 (+/- 0.022)|38.45% (+/- 1.7%)
|3 sec|0%|0.8444|0.5961|61%|
| |15%|0.8195 (+/- 0.023)|0.4723 (+/- 0.022)|50.4% (+/- 2.3%)
| |20%|0.7932 (+/- 0.025)|0.4146 (+/- 0.026)|46.9% (+/- 2.6%)
|5 sec|0%|0.8569|0.5862|56.7%|
| |15%|0.3458 (+/- 0.028)|0.039 (+/- 0.008)|6.2% (+/- 1.3%)
| |20%|0.348 (+/- 0.022)|0.041 (+/- 0.017)|6.9% (+/- 2.2%)

#### Performance (mean + std.) of ResNeSt

|Duration|% of frames removed|AUC|F1|Accuracy|
|:----:|:---------:|:-:|:----:|:------:|
|1 sec|0%|0.7346|0.4217|43.3%|
| |15%|0.734 (+/- 0.014)|0.4133 (+/- 0.016)|43.11% (+/- 1.45%)
| |20%|0.7524 (+/- 0.020)|0.4348 (+/- 0.018)|45.11% (+/- 2.4%)
|3 sec|0%|0.8301|0.4622|50%|
| |15%|0.7971 (+/- 0.016)|0.4358 (+/- 0.039)|47.22% (+/- 3.2%)
| |20%|0.7741 (+/- 0.016)|0.4008 (+/- 0.033)|44.2% (+/- 2.4%)
|5 sec|0%|0.8443|0.4739|51.1%|
| |15%|0.8115 (+/- 0.009)|0.4471 (+/- 0.031)|49.78% (+/- 2.9%)
| |20%|0.7596 (+/- 0.011)|0.4323 (+/- 0.049)|47.78% (+/- 4.1%)







