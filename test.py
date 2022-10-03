from train import mymodel
from data_process import *
import matplotlib.pyplot as plt
from packaging import version
import os
#matplotlib inline
plt.style.use("ggplot")


#test loss
loss,rmse,mae=mymodel.evaluate(x_close_test,y_close_test)

print(loss,rmse,mae)

predicted=mymodel.predict(x_close_test)
test_label=y_close_test.reshape(-1,1)
predicted=np.array(predicted[:,0]).reshape(-1,1)
len_train=len(x_close_train)

for j in range(len_train,len_train+len(x_close_test)):
    temp=df.iloc[j,4]
    predicted[j-len_train]=(predicted[j-len_train]+1)*temp
    test_label[j-len_train]=(test_label[j-len_train]+1)*temp


#real stock loss
plt.figure(2)
plt.plot(predicted, color = 'green', label = 'Predicted Price')
plt.plot(test_label, color = 'red', label = 'Real Price')
plt.title(sys.argv[2]+' Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
#plt.show()
plt.savefig(results_dir+sys.argv[2]+'.png')