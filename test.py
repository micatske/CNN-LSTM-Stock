from train import mymodel
from data_process import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
fig, ax = plt.subplots()
textstr = '\n'.join((
    r'$\mathrm{RMSE}=%.5f$' % (rmse),
     r'$\mathrm{MAE}=%.5f$' % (mae)))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.75, 0.1, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
plt.plot(predicted, color = 'green', label = 'Predicted Price')
plt.plot(test_label, color = 'red', label = 'Real Price')
plt.title(sys.argv[2]+' Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
#plt.show()
plt.savefig(results_dir+sys.argv[2]+'.png')