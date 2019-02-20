from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.externals import joblib
import sklearn
from sklearn import tree
import PIL

#READ LABELS
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, mean_absolute_error, \
    confusion_matrix


def getFeatures(num_of_imgs):
    model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(100, 100, 3), pooling='avg',
                  classes=1000)
    # model.compile()

    vgg16_feature_list = []
    img_path = 'b1-16-17/fig-'

    for i in range(0, num_of_imgs):
        path = img_path + str(i) + '.png'
        img = image.load_img(path, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        vgg16_feature = model.predict(x)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())

    feature_array = np.array(vgg16_feature_list)
    return feature_array


def saveFeatures(filename, farray):
    np.save(filename, farray)


def readFeatures(filename):
    return np.load(filename)

def make_data_for_redd(flist,vgg16list):
    label_List = []
    for n in flist:
        f = open(n).readlines()
        for line in f:
            label = line.split(' ')
            labelarr = np.asarray(label).astype(np.float)
            labelavg = np.average(labelarr)
            if (labelavg > 0.00270):
                labelavg = 1
            else:
                labelavg = 0
            label_List.append(labelavg)

    vgg16Arr= []
    # vgg16Arr.extend(vgg16list)

    for n in vgg16list:
        tmp = readFeatures(n)
        for i in range(0,tmp.__len__()):
            vgg16Arr.append(tmp[i])

    return (np.asarray(label_List), np.asarray(vgg16Arr))


def create_multilable_y(filenameA, filenameB, thressholdA, thressholdB):
    fA = open(filenameA)
    fB = open(filenameB)
    new_Y= []
    for l1, l2 in zip(fA, fB):
        l1 = l1.split(' ')
        l2 = l2.split(' ')
        l1 = np.asarray(l1).astype(np.float)
        l2 = np.asarray(l2).astype(np.float)
        avg1 = np.average(l1)
        avg2 = np.average(l2)
        if (avg1 > thressholdA):
            avg1 = 1
        else: avg1 = 0
        if (avg2 > thressholdB):
            avg2 = 1
        else: avg2 = 0
        new_Y.append([avg1, avg2])
    return np.asarray(new_Y)

def runTrainRedd(device):
    # REDD READING
    labelList, vgg16_feature_array  = make_data_for_redd(['data/'+device+'1-b1-labels', 'data/'+device+'1-b2-labels',
                                    'data/'+device+'1-b3-labels', 'data/'+device+'1-b4-labels',
                                    'data/'+device+'1-b5-labels', 'data/'+device+'1-b6-labels'],
                                   ['numpy-files/vgg16-redd-b1.npy', 'numpy-files/vgg16-redd-b2.npy',
                                    'numpy-files/vgg16-redd-b3.npy', 'numpy-files/vgg16-redd-b4.npy',
                                    'numpy-files/vgg16-redd-b5.npy','numpy-files/vgg16-redd-b6.npy'])
    num_of_imgs = labelList.__len__()

    train_X, test_X, train_Y, test_Y = train_test_split(vgg16_feature_array[:num_of_imgs], labelList, test_size=0.30,
                                                        random_state=42)

    # ##### Uncomment classifier of choice #####

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=500, learning_rate=0.25)

    # clf = AdaBoostClassifier(n_estimators=1000, learning_rate=0.25)

    # clf = AdaBoostClassifier(RandomForestClassifier(random_state=0.7), n_estimators=1000, learning_rate=0.5)

    clf = DecisionTreeClassifier(max_depth=15)

    # clf = RandomForestClassifier(n_estimators=1000, random_state=7)

    # clf = MLPClassifier(hidden_layer_sizes=500, batch_size=20)

    # Train classifier
    clf.fit(train_X,train_Y)

    # Save classifier for future use
    joblib.dump(clf, 'Tree'+'-'+device+'-redd-all.joblib')

    # Predict test data
    pred = clf.predict(test_X)

    # Print metrics
    printmetrics(test_Y,pred)

    return


def printmetrics(test, predicted):
    ##CLASSIFICATION METRICS

    f1m = f1_score(test, predicted, average='macro')
    f1 = f1_score(test, predicted)
    acc = accuracy_score(test, predicted)
    rec = recall_score(test, predicted)
    prec = precision_score(test, predicted)

    # print('f1:',f1)
    # print('acc: ',acc)
    # print('recall: ',rec)
    # print('precision: ',prec)

    # # to copy paste print
    print("=== For docs: {:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}".format(rec, prec, acc, f1m, f1))

    # ##REGRESSION METRICS
    # mae = mean_absolute_error(test_Y,pred)
    # print('mae: ',mae)
    # E_pred = sum(pred)
    # E_ground = sum(test_Y)
    # rete = abs(E_pred-E_ground)/float(max(E_ground,E_pred))
    # print('relative error total energy: ',rete)
    return


def plot_predicted_and_ground_truth(test, predicted):
    import matplotlib.pyplot as plt
    plt.plot(predicted.flatten(), label = 'pred')
    plt.plot(test.flatten(), label= 'Y')
    plt.show()
    return


def runTrainUkdale(device, house):
    # Read data labels
    h = house.split('-')
    print h[0]
    if h[0] == '1':
        f = open('data/'+device+h[1]+'-'+h[2]+'-labels').readlines()
    else:
        f = open('data/' + device + '1-1-labels').readlines()

    # Read thresholds
    thres = float(readThreshold(device, house))


    labelList = []
    for line in f:
        label = line.split(' ')
        labelarr = np.asarray(label).astype(np.float)
        labelavg = np.average(labelarr)
        if (labelavg > thres):
            labelavg = 1
        else:
            labelavg = 0
        labelList.append(labelavg)

    labelList = np.asarray(labelList)

    # UKDALE READING

    print('completed reading labels')

    num_of_imgs = labelList.__len__()

    # Uncomment below if needed to create own vgg16 feautures :
    # ---------------------------------------------------------
    # vgg16_feature_array = getFeatures(num_of_imgs-1)
    # saveFeatures('numpy-files/vgg16-b1-16-17.npy', vgg16_feature_array)
    # print('save completed')
    # ---------------------------------------------------------

    vgg16_feature_array = readFeatures('/home/nick/PycharmProjects/nanaproj/numpy-files/vgg16-b1-16-17.npy')
    # vgg16_feature_array = vgg16_feature_array[:labelList.__len__()]
    train_X, test_X, train_Y, test_Y = train_test_split(vgg16_feature_array[:num_of_imgs], labelList[:num_of_imgs-1], test_size=0.99,
                                                        random_state=42)

    # clf = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=5, learning_rate=0.5)

    # clf = RandomForestRegressor(n_estimators=10, random_state=7)

    # clf = MLPRegressor(hidden_layer_sizes=20, activation='tanh')

    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=500, learning_rate=0.25)

    # clf = AdaBoostClassifier(n_estimators=1000, learning_rate=0.25)

    # clf = AdaBoostClassifier(RandomForestClassifier(random_state=0.7), n_estimators=1000, learning_rate=0.5)

    # clf = DecisionTreeClassifier(max_depth=15)

    # clf = RandomForestClassifier(n_estimators=1000, random_state=7)

    # clf = MLPClassifier(hidden_layer_sizes=500, batch_size=20)

    # cv = cross_val_score(model_tree, train_X, train_Y, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (cv.mean(), cv.std() * 2))
    #
    # clf.fit(train_X,train_Y)
    # joblib.dump(clf, 'MLP5-dishwasher-redd-all.joblib')
    clf = joblib.load('/media/nick/Ext hard dr/NILM nana/models/AdaTree1000-washingmachine-13-14-b1.joblib')
    pred = clf.predict(test_X)
    # #
    # # confMatrix = confusion_matrix(test_Y, pred)
    # # print("confusion matrix: ", confMatrix)
    #
    # # metrics
    printmetrics(test_Y, pred)
    #
    plot_predicted_and_ground_truth(test_Y, pred)
    return


def readThreshold(device, house):
    threshold = 0
    f = open('thresholds-'+device+'.txt').readlines()
    for line in f:
        splittedline = line.split(',')
        if splittedline[0] == house:
            threshold = splittedline[1]
    return threshold


runTrainUkdale('washing machine', '1-16-17')

