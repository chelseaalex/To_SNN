因为MAE是重建，不是分类，于是加了分类头后在cifar数据集上训练，训练的效果为94.06：
<img width="1447" alt="Vit_MAE" src="https://github.com/chelseaalex/anyi/assets/71577910/b7d91a97-228f-4272-9f23-7ade6830aee0">

而在转化为SNN后的分类精度可以达到96.37，证明了SNN的优良效果：
<img width="1375" alt="Vit_MAE_SNN" src="https://github.com/chelseaalex/anyi/assets/71577910/332874fb-6965-4bdf-b24b-8b4a2162638b">
