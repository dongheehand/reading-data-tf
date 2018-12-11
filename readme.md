## Example code to measure the performance of tf.data API 
The example code predicts the class label of input image by using [VGG19](https://arxiv.org/pdf/1409.1556.pdf) network

#### For experiments
You should download pre-trained VGG19 weight [vgg19.npy](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

1) If you want to use tf.data API, pipe_lining option to True
2) If you want to read data with main memory, in_memory option to True
3) If you want to read data without main memory (on disk), in_memory option to False

#### Experimental Results
I measured the average time for predicting class label.

GPU : Nvidia Tesla K80

CPU : Intel(R) Xeon(R) CPU E5-2686 v4@2.30GHz

Batch size : 32

|| With tf.data API  | w.o tf.data API(using feed_dict) |
|------| ------------- | ------------- |
|in_memory| 0.3459s | 0.3410s  |
|disk based| 1.4656s  | 4.0910s  |



### Comments
If you have any questions or comments on my codes, please email to me. [son1113@snu.ac.kr](mailto:son1113@snu.ac.kr)