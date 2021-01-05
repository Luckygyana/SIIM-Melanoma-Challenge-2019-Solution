# SIIM-Melanoma-Challenge-2019-Solution
Congratatulations to the winners, and thanks to kaggle, competition sponsor and kernel contributors who shared their insights, it helped us a lot. Congratulations to my teammates @masterroshi @upbeatfreak

Our current position standing 6th Private (23rd Public) is totally because of @cdeotte amazing contribution.
Our Approach
We started this competition after 2 months after the start.For a baseline, we used @cdeotte 's Triple Stratified Kfold With Tfrecords for tensorflow and @shonenkov 's [Training CV] Melanoma Starter. thankyou for these amazing kernels.we started working on them.

We used EfficientNet [B0-B6], Resnest,Resnext, with Sizes 192x192 256x256 384x384 512x512 768x768 384x512[HxW]

Summary
What Worked for Us
Heavy TTA (X20)
Cutmix
coarse dropout
SWA
loss-Label Smoothing
optimizer-AdamW, Adam
BCE
2018, 2020 and malignant datasets
5 checkpoints' prediction averaging(stabalised our model's predictions)
some models were trained with different height width ratios

What didn't Work for Us
loss functions-Focal loss, dice loss
optimizer- Ranger
hair removal/addition
pseudo labelling
2019 dataset
preprocessing techniques from aptos competition
progressive learning

Ensembling techniques
weighted average
Power Average
minmax ensemble(didn't help)
3hr before end of competition I came across rank ensembling and and we did this ensemble and got 0.9697 for our last submission

Our 3 final Submission
we new the shakeup was coming, so we tried to select different approaches

All pytorch gpu solution models(with context) - 0.9530 (public LB) 0.9380 (private LB) 0.9541 (CV)
All pytorch model (with context) and All tf models (without context) - 0.9627 (public LB) 0.9470 (private LB) 0.9618 (CV)
Blend of public submission with 2nd submission with post proccessing technique - 0.9697 (public LB) 0.9126 (private LB) (overfitted)
all the above were also ensembled with the meta only submission.
we wanted to give a shot to public lb overfitted submission but obviosly didn't work out well.
I guess we were lucky enough to select the best private lb submission from our arsenel

we found the discussions and public kernels really fruitful and learnt a lot from this competition.

We are quite new in kaggle and this is our first competition. We will share the code in some days.
