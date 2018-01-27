# Guo
在找数据集时尽量多找不同的图片，存到对应的文件夹下就可以，因为有路径问题，文件夹的名字最好不要改
在训练数据集时，如果在前几次epoch中acc就已经比较高，就把epoch改小一点
有两种训练方法，VGG的方法应该更好一些
训练完成后找一个精度最高的h5文件，手动加入到predict.py中，预测时按c是拍照然后将预测值输出
数据集可以自己拍，用datacollection.py按下c连续拍照，按下esc退出

上传的这个h5精度并不高，因为当时运行的是validation文件夹下面的，现在如果再运行应该是train文件夹下面的，但是预测训练集中的图片应该都准确

这是一整张图片分割出手的部分，用的是Haar训练的级联分类器

目前图片数量少还不准确

训练的时候，将只包含人手的照片放到posdata文件夹中，然后双击bat文件，会生成一个txt文件，将文件内容的格式换成现在文件的格式，然后将文件内容复制到posdata中，注意posdata文件夹下面的图片必须是20*20，可以使用handcollection.py自己拍照片，然后输入命令opencv_createsamples.exe -info  posdata.txt -vec pos.vec -num posdata文件夹下真实的图片数量 -w 20 -h 20

在negdata中放不包含手的图片，应该最好是黑色背景，可以加入脸看一下效果，在background.rar中有我的脸，然后同样双击bat格式不用替换，只包含文件名就行，最后一定不要有空行，也不用执行上面的命令,可以通过backgroundcollection.py拍照

最后输入命令
opencv_haartraining.exe -data xml -vec pos.vec -bg negdata.txt -npos 正样本数量 -nneg 负样本数量 -nstages 4 -nsplits 1 -mem 1280 -nonsym -w 20 -h 20

nstages代表轮数，应该是越大越准确，但是太大也能会死掉，注意正负样本比例为1:3左右，负样本不能太少，否则也会死掉

注意每次训练先删掉xml文件夹下面的所有文件

最后可以通过detectHand.py测试
