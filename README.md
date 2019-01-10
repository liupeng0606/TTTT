## Sentence Pair Matching
# model summary
```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 80)           0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 100)          0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 80, 100)      88100       input_1[0][0]                    
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 100, 100)     242500      input_2[0][0]                    
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 80, 200)      160800      embedding_1[0][0]                
__________________________________________________________________________________________________
bidirectional_2 (Bidirectional) (None, 100, 200)     160800      embedding_2[0][0]                
__________________________________________________________________________________________________
attention_layer_1 (Attention_la (None, 200)          40200       bidirectional_1[0][0]            
__________________________________________________________________________________________________
attention_layer_2 (Attention_la (None, 200)          40200       bidirectional_2[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 400)          0           attention_layer_1[0][0]          
                                                                 attention_layer_2[0][0]          
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 100)          40100       concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 3)            303         dense_1[0][0]                    
==================================================================================================
```

# log information

```
Epoch 1/20
256/256 [==============================] - 8s 30ms/step - loss: 1.2982 - acc: 0.3984 - val_loss: 1.2411 - val_acc: 0.2344
Epoch 2/20
256/256 [==============================] - 3s 11ms/step - loss: 1.1531 - acc: 0.3164 - val_loss: 1.1367 - val_acc: 0.3281
Epoch 3/20
256/256 [==============================] - 3s 11ms/step - loss: 1.0855 - acc: 0.4141 - val_loss: 1.1955 - val_acc: 0.2969
Epoch 4/20
256/256 [==============================] - 3s 11ms/step - loss: 1.0521 - acc: 0.4961 - val_loss: 1.0998 - val_acc: 0.3750
Epoch 5/20
256/256 [==============================] - 3s 11ms/step - loss: 0.9981 - acc: 0.5156 - val_loss: 1.1659 - val_acc: 0.3594
Epoch 6/20
256/256 [==============================] - 3s 11ms/step - loss: 0.9088 - acc: 0.5781 - val_loss: 1.1275 - val_acc: 0.4688
Epoch 7/20
256/256 [==============================] - 3s 11ms/step - loss: 0.8534 - acc: 0.5977 - val_loss: 1.1936 - val_acc: 0.5312
Epoch 8/20
256/256 [==============================] - 3s 11ms/step - loss: 0.7819 - acc: 0.6562 - val_loss: 1.1795 - val_acc: 0.4844
Epoch 9/20
256/256 [==============================] - 3s 11ms/step - loss: 0.7425 - acc: 0.6523 - val_loss: 1.1783 - val_acc: 0.5625
Epoch 10/20
256/256 [==============================] - 3s 11ms/step - loss: 0.6947 - acc: 0.7070 - val_loss: 1.2424 - val_acc: 0.5312
Epoch 11/20
256/256 [==============================] - 3s 11ms/step - loss: 0.6766 - acc: 0.7031 - val_loss: 1.0734 - val_acc: 0.5625
Epoch 12/20
256/256 [==============================] - 3s 11ms/step - loss: 0.6296 - acc: 0.7266 - val_loss: 1.1043 - val_acc: 0.5938
Epoch 13/20
256/256 [==============================] - 3s 11ms/step - loss: 0.5670 - acc: 0.7617 - val_loss: 1.2102 - val_acc: 0.6250
Epoch 14/20
256/256 [==============================] - 3s 11ms/step - loss: 0.4967 - acc: 0.8047 - val_loss: 1.1464 - val_acc: 0.6250
Epoch 15/20
256/256 [==============================] - 3s 11ms/step - loss: 0.5594 - acc: 0.7812 - val_loss: 1.1649 - val_acc: 0.5938
Epoch 16/20
256/256 [==============================] - 3s 11ms/step - loss: 0.4682 - acc: 0.8047 - val_loss: 1.3050 - val_acc: 0.5938
Epoch 17/20
256/256 [==============================] - 3s 11ms/step - loss: 0.4777 - acc: 0.8242 - val_loss: 1.2638 - val_acc: 0.6250
Epoch 18/20
256/256 [==============================] - 3s 11ms/step - loss: 0.4110 - acc: 0.8398 - val_loss: 1.4328 - val_acc: 0.5781
Epoch 19/20
256/256 [==============================] - 3s 11ms/step - loss: 0.3088 - acc: 0.8789 - val_loss: 1.3981 - val_acc: 0.6094
Epoch 20/20
256/256 [==============================] - 3s 11ms/step - loss: 0.3176 - acc: 0.8711 - val_loss: 1.5595 - val_acc: 0.5156
```
