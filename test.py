from train import mymodel
from data_process import *
import matplotlib.pyplot as plt
#matplotlib inline
plt.style.use("ggplot")


#test loss
loss,rmse,mae=mymodel.evaluate(x_close_test,y_close_test)
plt.figure(3)
plt.plot(loss,color = 'red', label = 'test')
plt.legend()
plt.show()
print(loss)

predicted=mymodel.predict(x_close_test)
test_label=y_close_test.reshape(-1,1)
predicted=np.array(predicted[:,0]).reshape(-1,1)
len_train=len(x_close_train)

for j in range(len_train,len_train+len(x_close_test)):
    temp=df.iloc[j,3]
    predicted[j-len_train]=(predicted[j-len_train]+1)*temp
    test_label[j-len_train]=(test_label[j-len_train]+1)*temp


#real stock loss
plt.figure(2)
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()