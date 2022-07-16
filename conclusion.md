I did four kinds of experiments on sentence representation.
1. one hiddden layer (128 neurons) with 1536-dim input features
2. three hidden layers (128 nerons each layer) with 1536-dim input features
3. one hiddden layer (128 neurons) with 3072-dim input features
4. three hidden layers (128 nerons each layer) with 3072-dim input features

Based on results, I have two findings:
1. Adding more layers/ make classifier more complex doesn't significantly help to improve accuracy
2. More input features help to boost the performance

conclusion 1: Fixed layer size, and change vector length, still found more features works better. If input features are always 3072, there is no great difference between single layer and multiple layers.

conclusion 2: Vectors of higher dimensions boost accuracy for 2-3%, and most optimal models are single-layer. 

I also took a look at the training details and found if input feature is 1536-dims (simply concatenate two sentence vectors), training accuracy is 0.48 for maximum. But 3072 dims (with element-wise operations) lead to a higher value (0.54). 6% higher.


