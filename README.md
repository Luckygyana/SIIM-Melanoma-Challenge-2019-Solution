# [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) (**6th rank approch**)

### Identify melanoma in lesion images

## Team Members

- [Pratik Bedre](https://www.kaggle.com/cdeotte)
- [Gyanendra Das](https://github.com/Luckygyana)
- [Saksham Aggarwal](https://github.com/saksham20aggarwal)

# Our Approach

We started this competition after 2 months after the start. For a baseline, we used [@cdeotte](https://www.kaggle.com/cdeotte)'s [Triple Stratified Kfold With Tfrecords](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords) for tensorflow and @[shonenkov](https://www.kaggle.com/shonenkov) 's [Training CV Melanoma Starter](https://www.kaggle.com/shonenkov/training-cv-melanoma-starter) for PyTorch.
Thanks to these amazing kernels.

We used **EfficientNet [B0-B6]**, **Resnest**,**Resnext**, with Sizes **192x192** **256x256** **384x384** **512x512** **768x768** **384x512**[HxW]

## Summary

#### What Worked for Us

- Heavy TTA (X20)
- Cutmix
- Coarse dropout
- SWA(Stochastic Weight Averaging)
- Loss-Label Smoothing, BCE
- Optimizers - AdamW, Adam
- 2018, 2020 and malignant datasets
- 5 checkpoints' prediction averaging(stabalised our model's predictions)
- some models were trained with different height width ratios

#### What didn't Work for Us

- Loss functions-Focal loss, dice loss
- Optimizer- Ranger
- Hair removal/addition
- Pseudo labelling
- 2019 dataset
- Preprocessing techniques from [Aptos Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Progressive learning

### Ensembling techniques

- Weighted average
- Power Average
- Minmax ensemble(didn't help)<br>
<br>3hr before end of competition we came across rank ensembling and and we did this ensemble and got **0.9697** for our last submission

## Our 3 final Submission

We new the shakeup was coming, so we tried to select different approaches.

1.  All 15+ pytorch gpu solution models(with context) - 0.9530 (public LB) 0.9380 (private LB) 0.9541 (CV)
2.  15+ pytorch model (with context) and 15+ tf models (without context) - 0.9627 (public LB) 0.9470 (private LB) 0.9618 (CV)
3.  Blend of public submission with 2nd submission with post proccessing technique - 0.9697 (public LB) 0.9126 (private LB) (overfitted)
all the above were also ensembled with the meta only submission.
<br>
    We wanted to give a shot to public lb overfitted submission but obviosly didn't work out well.
    I guess we were lucky enough to select the best private lb submission from our arsenel.
    We found the discussions and public kernels really fruitful and learnt a lot from this competition.

