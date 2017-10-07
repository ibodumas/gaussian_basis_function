#################### This cell is for all the methods and data import ################################
import os
import sklearn.model_selection
import pandas
from numpy.linalg import inv  # matrix inverse
import random  # for seeding and random no generation
from itertools import chain  # for unlisting
import matplotlib.pyplot as matplot
from numpy import linalg
import numpy

# stratification
def folds_stratify(nSample, seed, kFolds):  # this return kfold stratification
    random.seed(seed)
    foldSize = round(nSample / kFolds, 0)
    randomList = list(range(0, nSample))
    random.shuffle(randomList)
    stra = []

    for k in range(0, kFolds):
        strt = int(k * foldSize)
        end = int((k + 1) * foldSize)
        if k == (kFolds - 1):
            end = nSample
        stra.append(list(randomList)[strt:end])

    return stra
# end def folds_stratify

# phi for Gaussian basis function train and test #same centres have to be used
def phi_gbf(x_train, x_test, nCentre, scale):
    centre_loc = list(range(0, nCentre))  # location of the centres
    random.shuffle(centre_loc)  # shuffle the centres
    nRow = x_train.shape[0]
    nRow_test = x_test.shape[0]
    X_p = pandas.DataFrame(x_train)
    X_p_test = pandas.DataFrame(x_test)
    phi_train = pandas.DataFrame(index=list(range(0, nRow)), columns=list(range(0, nCentre)))
    phi_test = pandas.DataFrame(index=list(range(0, nRow_test)), columns=list(range(0, nCentre)))
    centre = None

    for index, pos in enumerate(centre_loc):
        if pos >= x_train.shape[0]:
            randomList = list(range(0, x_train.shape[0]))
            random.shuffle(randomList)
            pos = randomList[0]
            centre = numpy.asarray(X_p.iloc[pos]) + list(chain.from_iterable(0.001*numpy.random.rand(1, X_p.shape[1]))) #ADDING NOISE to avoid same col or row
        else:
            centre = numpy.asarray(X_p.iloc[pos])

        centre = numpy.asarray(centre)
        dis = linalg.norm(X_p - centre, axis=1)
        phi_cell = numpy.exp(-numpy.square(dis) / (2 * numpy.square(scale)))
        phi_train.iloc[:, index] = phi_cell
        dis_test = linalg.norm(X_p_test - centre, axis=1)
        phi_cell_test = numpy.exp(-numpy.square(dis_test) / (2 * numpy.square(scale)))
        phi_test.iloc[:, index] = phi_cell_test
    return [phi_train, phi_test]
#end phi_gbf

def W_gbf(phi, y, lamda):  # linear basis function# lamda = reqularization coefficient
    phi_p = pandas.DataFrame(phi)
    X = pandas.concat([phi_p[0], phi_p], axis=1)  # Adding one column to X
    X.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
    X = numpy.asarray(X)
    phi = X  # in linear basis function #ϕ
    phi_trans = phi.transpose()  # phi transpose
    phi_trans_phi = numpy.dot(phi_trans, phi)
    I = numpy.identity(phi_trans_phi.shape[0])  # add 1 to nFeature bcos of x0
    lamda_I = lamda * I
    add_lamda_I_phi_trans_phi = numpy.add(lamda_I, phi_trans_phi)
    inv_sum = inv(add_lamda_I_phi_trans_phi)  # inverse the matrix above
    inv_sum_phi_trans = numpy.dot(inv_sum, phi_trans)
    W = numpy.dot(inv_sum_phi_trans, y)
    return W
#end W_gbf

# cross validation for gaussian basis function
def cv_gbf(data, noutputs, kFolds, lamda, width, centre, seed):
    nFeature = data.shape[1] - noutputs
    nSample = data.shape[0]
    stra_all = folds_stratify(nSample=nSample, seed=seed, kFolds=kFolds)
    columns = None
    if noutputs == 1:
        columns = ["Combinations of: lamda, s, u", "error-y0"]
    elif noutputs == 3:
        columns = ["Combinations of: lamda, s, u", "error-y0" , "error-y1", "error-y2"]
    #list(range(0, noutputs + 1))
    df = pandas.DataFrame(index=list(range(0, len(lamda) * len(width) * len(centre))), columns=columns)

    n = 0
    while n < noutputs:
        r = 0
        for lamda_ind, lamda_val in enumerate(lamda):
            for width_ind, width_val in enumerate(width):
                for centre_ind, centre_val in enumerate(centre):
                    df.iloc[r, 0] = [lamda_val, width_val, centre_val]
                    seed = 1209345 + centre_ind #maintaining the same centre locations for each y(s)
                    k = 0
                    err_folds = []
                    while k < kFolds:
                        stra = stra_all.copy()
                        test = data[stra[k]]
                        del stra[k]  # del test list
                        stra_train = list(chain.from_iterable(stra))  # merge the sublists
                        train = data[stra_train]
                        train_x = train[:, 0:nFeature]
                        train_y = train[:, nFeature + n]
                        test_x = test[:, 0:nFeature]
                        y_actual = test[:, nFeature + n]
                        y_actual.shape = (y_actual.shape[0], 1)
                        phi_gbf_total = phi_gbf(x_train=train_x, x_test=test_x, nCentre=centre_val, scale=width_val)
                        phi_train = phi_gbf_total[0]
                        phi_test = phi_gbf_total[1]
                        w_vals = W_gbf(phi=phi_train, y=train_y, lamda=lamda_val)
                        #calculating error
                        #print("crazy smart {}".format([lamda_val, width_val, centre_val]))
                        w_vals.shape = (w_vals.shape[0], 1)
                        phi_test = pandas.DataFrame(phi_test)
                        phi_test = pandas.concat([phi_test[0], phi_test], axis=1)  # Adding one column to X
                        phi_test.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
                        phi_test = numpy.asarray(phi_test)
                        y_pred = numpy.dot(phi_test, w_vals)
                        if y_actual.shape != y_pred.shape:
                            print("\n\nError003: Shape not equal: y_actual.shape != y_pred.shape\n\n")
                        y_actual_pred = numpy.subtract(y_actual, y_pred)
                        error_2 = numpy.square(y_actual_pred)
                        errors = numpy.sum(error_2, axis=0)
                        err_folds.append(errors)
                        k += 1

                    aver_err_fold = sum(err_folds) / len(err_folds)
                    df.iloc[r, n+1] = aver_err_fold
                    r += 1
        n += 1
    return df
# end def cv_gbf

def cv_gbf_all_data(data, noutputs, kFolds, lamda, width, centre, seed):
    nFeature = data.shape[1] - noutputs
    nSample = data.shape[0]
    stra_all = folds_stratify(nSample=nSample, seed=seed, kFolds=kFolds)
    n = 0
    errors_per_y = []
    while n < noutputs:
        error_per_fold = []
        lamda_ = lamda[n]
        width_ = width[n]
        centre_ = centre[n]
        k = 0
        while k < kFolds:
            stra = stra_all.copy()
            test = data[stra[k]]
            del stra[k]  # del test list
            stra_train = list(chain.from_iterable(stra))  # merge the sublists
            train = data[stra_train]
            y_actual = test[:, nFeature + n]
            y_actual.shape = (y_actual.shape[0], 1)
            train_x = train[:, 0:nFeature]
            train_y = train[:, nFeature + n]
            test_x = test[:, 0:nFeature]
            y_actual = test[:, nFeature + n]
            y_actual.shape = (y_actual.shape[0], 1)
            phi_gbf_total = phi_gbf(x_train=train_x, x_test=test_x, nCentre=centre_, scale=width_)
            phi_train = phi_gbf_total[0]
            phi_test = phi_gbf_total[1]
            w_vals = W_gbf(phi=phi_train, y=train_y, lamda=lamda_)
            # calculating error
            w_vals.shape = (w_vals.shape[0], 1)
            phi_test = pandas.DataFrame(phi_test)
            phi_test = pandas.concat([phi_test[0], phi_test], axis=1)  # Adding one column to X
            phi_test.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
            phi_test = numpy.asarray(phi_test)
            y_pred = numpy.dot(phi_test, w_vals)
            if y_actual.shape != y_pred.shape:
                print("\n\nError003: Shape not equal: y_actual.shape != y_pred.shape\n\n")
            y_actual_pred = numpy.subtract(y_actual, y_pred)
            error_2 = numpy.square(y_actual_pred)
            errors = numpy.sum(error_2, axis=0)
            error_per_fold.append(errors)
            k += 1
        n += 1
        errors_per_y.append(numpy.mean(error_per_fold))
    return [errors_per_y, numpy.sum(errors_per_y)]

#validation for gaussian basis function
def gbf_main(train, test, noutputs, lamda, width, centre):
    nFeature = train.shape[1] - noutputs
    columns = None
    if noutputs == 1:
        columns = ["error-y0"]
    elif noutputs == 3:
        columns = ["error-y0" , "error-y1", "error-y2"]
    #list(range(0, noutputs + 1))
    df = pandas.DataFrame(index=["error"], columns=columns)
    actual_pred = []
    err = []

    n = 0
    while n < noutputs:
        lamda_ = lamda[n]
        width_ = width[n]
        centre_ = centre[n]
        df_actual_pred = pandas.DataFrame(index=list(range(0, test.shape[0])), columns=["actual", "predict"])
        train_x = train[:, 0:nFeature]
        train_y = train[:, nFeature + n]
        test_x = test[:, 0:nFeature]
        y_actual = test[:, nFeature + n]
        y_actual.shape = (y_actual.shape[0], 1)
        phi_gbf_total = phi_gbf(x_train=train_x, x_test=test_x, nCentre=centre_, scale=width_)
        phi_train = phi_gbf_total[0]
        phi_test = phi_gbf_total[1]
        w_vals = W_gbf(phi=phi_train, y=train_y, lamda=lamda_)
        # calculating error
        w_vals.shape = (w_vals.shape[0], 1)
        phi_test = pandas.DataFrame(phi_test)
        phi_test = pandas.concat([phi_test[0], phi_test], axis=1)  # Adding one column to X
        phi_test.iloc[:, 0] = 1  # setting x0 = 1, from the column added above
        phi_test = numpy.asarray(phi_test)
        y_pred = numpy.dot(phi_test, w_vals)
        if y_actual.shape != y_pred.shape:
            print("\n\nError003: Shape not equal: y_actual.shape != y_pred.shape\n\n")
        y_actual_pred = numpy.subtract(y_actual, y_pred)
        error_2 = numpy.square(y_actual_pred)
        errors = numpy.sum(error_2, axis=0)
        err.append(errors)
        df_actual_pred["actual"] = y_actual
        df_actual_pred["predict"] = y_pred
        actual_pred.append(df_actual_pred)

        print("\n" + "Summary table of test data relating to y{}\n".format(n) + df_actual_pred.head(5).to_string() + "\n")
        # plot
        print("y_actual vs. predict for variable y{} \n".format(n))
        matplot.scatter(y_actual, y_pred)
        matplot.xlabel('y_actual_gbf')
        matplot.ylabel('y_pred_gbf')
        matplot.show()
        n += 1

    finalError = sum(err)
    return finalError
# end def cv_gbf

# my_regression
def my_regression(trainX, testX, noutputs):
    columns = None
    row = ["best_params", "best_error"]
    if noutputs == 1:
        columns = ["y0"]
    elif noutputs == 3:
        columns = ["y0", "y1", "y2"]

    nFeature = trainX.shape[1] - noutputs  # No of features

    #Regularised guassian basis function####################################################
    #CV on these sets of lamda, width, and centre; to determine the best parameters
    seed_gbf = 3745193
    kFolds_gbf = 5
    lamdas_gbf = [0, 0.01]
    widths_gbf = [1, 5, 10]
    centres_gbf = [10, 100]
    cv_gbf_best_param = cv_gbf(data=trainX, noutputs=noutputs, kFolds=kFolds_gbf, lamda=lamdas_gbf, width=widths_gbf, centre=centres_gbf, seed=seed_gbf)
    print("\n5-Folds Cross Validation\nTable of average error per parameters combinations per the target variable(s)" + "[lamda, width, centres]\n" + cv_gbf_best_param.to_string() + "\n")
    params_gbf_cv = cv_gbf_best_param.iloc[:, 0]
    df_gbf = pandas.DataFrame(index=row, columns=columns)
    total_error = []

    n = 0
    lamdas_ = []
    widths_ = []
    centres_ = []
    while n < noutputs:
        error_gbf_cv = list(chain.from_iterable(cv_gbf_best_param.iloc[:, n + 1]))
        minerr = numpy.min(error_gbf_cv)
        total_error.append(minerr)
        index_best_param = error_gbf_cv.index(minerr)
        best_params = params_gbf_cv[index_best_param]
        df_gbf.iloc[0, n] = best_params
        df_gbf.iloc[1, n] = minerr
        lamdas_.append(best_params[0])
        widths_.append(best_params[1])
        centres_.append(best_params[2])
        print("\n" + "The best sets of parameters for y{} are".format(n))
        print("lamda = {}".format(best_params[0]))
        print("width(s) = {}".format(best_params[1]))
        print("No of centers = {}".format(best_params[2]) + "\n")
        n += 1

    print("\nThe total error across all target variable(s) = {}".format(sum(total_error)))

    print("\nHaving chosen the best set of params:")
    print("\nBelow are the analysis of training the best parameters on trainX and evaluating on testX:")
    gbf_use_best_param = gbf_main(train=trainX, test=testX, noutputs=noutputs, lamda=lamdas_, width=widths_, centre=centres_)
    print("\n########## End of Gaussian Basis Function ############\n")
    # end gaussian basis function#############################################################
    return [df_gbf, [lamdas_, widths_, centres_]]
# end def my_regression

#CV for all dataset
def CV_on_all_data_gbf(allData, gbf_param, noutputs, dataName):
    kFolds = 5
    seed_gbf = 7587930
    cv_all_data_gbf = cv_gbf_all_data(data=allData, noutputs=noutputs, kFolds=kFolds, lamda=numpy.asarray(gbf_param[0]), width=numpy.asarray(gbf_param[1]), centre=numpy.asarray(gbf_param[2]), seed=seed_gbf)
    print("Gaussian BF ERROR for " + dataName + " = {}".format(cv_all_data_gbf[1]))
    return [cv_all_data_gbf]

####################################### Import Data #########################################
os.chdir('C:/Users/2PAC/Documents/Python Scripts/pycharm/ML/HW1_Regression') #set new directory
def z_score_norm(data):
    if type(data) is numpy.ndarray:
        mean = numpy.mean(data, axis=0)
            #data.mean()
        std = numpy.std(data, axis=0) #data.std()
        data_norm = (data - mean) / std
        result = data_norm
    else:
        result = "Error001: Provide numpy array"

    return result

'''AIRFOIL'''
#from numpy import loadtxt
airfoil = numpy.loadtxt("airfoil_self_noise.dat.txt")
sample_size_af = airfoil.shape[0]
airfoil_norm = z_score_norm(airfoil)
random.seed(5054123) #set seed
x_train_af, x_test_af = sklearn.model_selection.train_test_split(airfoil_norm, test_size=0.2, random_state=0)
noutputs_af = 1

'''YACHT'''
yacht = numpy.loadtxt("yacht_hydrodynamics.data.txt")
sample_size_yt = yacht.shape[0]
yacht_norm = z_score_norm(yacht)
random.seed(3452332)
x_train_yt, x_test_yt = sklearn.model_selection.train_test_split(yacht_norm, test_size=0.2, random_state=0)
noutputs_yt = 1

'''SLUMP'''
slump = numpy.loadtxt("slump_test.data.txt", skiprows=1, delimiter=",")
slump = slump[:,1:11]
sample_size_sp = slump.shape[0]
slump_norm = z_score_norm(slump)
random.seed(3450423)
x_train_sp, x_test_sp = sklearn.model_selection.train_test_split(slump_norm, test_size=0.2, random_state=0)
noutputs_sp = 3
####################################### End Import Data #########################################

#################### Airfoil Data - Linear Basis Funcition And Gaussian Basis Function ################################
myReg_airfoil = my_regression(trainX=x_train_af, testX=x_test_af, noutputs=noutputs_af)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL AIRFOIL DATA: RESULT")
cv_all_data_airfoil = CV_on_all_data_gbf(allData=airfoil_norm, gbf_param=myReg_airfoil[1], noutputs=noutputs_af, dataName="Airfoil")
#################### End of Airfoil Data  ################################

#################### Yacht Data - Linear Basis Funcition And Gaussian Basis Function ################################
myReg_yacht = my_regression(trainX=x_train_yt, testX=x_test_yt, noutputs=noutputs_yt)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL YACHT DATA: RESULT")
cv_all_data_yacht = CV_on_all_data_gbf(allData=yacht_norm, gbf_param=myReg_yacht[1], noutputs=noutputs_yt, dataName="Yacht")
#################### End of Yacht Data  ################################


#################### Slump Data - Linear Basis Funcition And Gaussian Basis Function ################################
myReg_slump = my_regression(trainX=x_train_sp, testX=x_test_sp, noutputs=noutputs_sp)
print("\nCROSS VALIDATION OUTSIDE my_Regrssion FUNCTION\n5 FOLDS CROSS VALIDATION FOR ALL SLUMP DATA: RESULT")
cv_all_data_slump = CV_on_all_data_gbf(allData=slump_norm, gbf_param=myReg_slump[1], noutputs=noutputs_sp, dataName="Slump")
#################### End of Slump Data  ################################
