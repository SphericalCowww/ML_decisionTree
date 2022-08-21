import sys, os, time
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt


def main():
    yColName = "Z";
    fileNameTrain = os.getcwd() + "/input2DPlotTrain.txt";
    fileNameTest  = os.getcwd() + "/input2DPlotTest.txt";
#training
    dfTrain = pd.read_csv(fileNameTrain, sep=",", skiprows=1, header=None);
    with open(fileNameTrain, "r") as fileRead:    
        columnStr = fileRead.readline().rstrip("\n");
        dfTrain.columns = columnStr.split(",");
    tree = buildTree(dfTrain, yColName);
    print("Training data:");
    print(dfTrain.columns.tolist());
    print(*dfTrain.values.tolist(), sep="\n");
    print("");
    print("Built tree:");
    printTree(tree);
    print("");
#predicting
    ''' 
    dfTest = pd.read_csv(fileNameTest, sep=",", skiprows=1, header=None);
    with open(fileNameTest, "r") as fileRead:
        columnStr = fileRead.readline().rstrip("\n");
        dfTest.columns = columnStr.split(",");
    if np.array_equal(dfTest.columns.to_numpy(),dfTrain.columns.to_numpy())==False:
        print("ERROR: incompatible training and testing data.");
        exit(1);
    data = predictY(tree, dfTest, verbosity=1);
    print("Predicted Data:");
    print(dfTest.columns.tolist());
    print(data); 
    '''
#plotting
    xRange = 5.0;
    yRange = 5.0;

    data = dfTrain.values.tolist(); 
    type0Pts = [row for row in data if row[2] == 0];
    type1Pts = [row for row in data if row[2] == 1];
    x0 = listCol(type0Pts, 0);
    y0 = listCol(type0Pts, 1);
    x1 = listCol(type1Pts, 0);
    y1 = listCol(type1Pts, 1);
    ax = plt.figure().add_subplot(111);
    plt.scatter(x0, y0, color="red", marker="o");
    plt.scatter(x1, y1, color="blue", marker="x");
    plt.title("Decision Tree Test", fontsize=24, y=1.03);
    plt.xlabel("x-value", fontsize=18);
    plt.ylabel("y-value", fontsize=18);
    plt.xlim(0, xRange);
    plt.ylim(0, yRange);

    region = [];
    for i, node in enumerate(tree):
        region.append({"index":i, "xLower":0, "xUpper":1, "yLower":0, "yUpper":1});
        parentIdx = node["parent"];
        region[i]["xLower"] = 1.0*region[parentIdx]["xLower"]; 
        region[i]["xUpper"] = 1.0*region[parentIdx]["xUpper"];
        region[i]["yLower"] = 1.0*region[parentIdx]["yLower"];
        region[i]["yUpper"] = 1.0*region[parentIdx]["yUpper"];
        if tree[parentIdx]["xAttrOpt"] == "X":
            if node["nodeDir"] == "left":
                region[i]["xUpper"] = tree[parentIdx]["xLeftOpt"]/xRange;
            elif node["nodeDir"] == "right":
                region[i]["xLower"] = tree[parentIdx]["xLeftOpt"]/xRange;
        elif tree[parentIdx]["xAttrOpt"] == "Y":
            if node["nodeDir"] == "left":
                region[i]["yUpper"] = tree[parentIdx]["xLeftOpt"]/yRange;
            elif node["nodeDir"] == "right":
                region[i]["yLower"] = tree[parentIdx]["xLeftOpt"]/yRange;
    print(*region, sep="\n");
    for i, node in enumerate(tree):
        if node["isLeaf"] == "No":
            if node["xAttrOpt"] == "X":
                plt.axvline(x=node["xLeftOpt"], \
                            ymin=region[i]["yLower"], ymax=region[i]["yUpper"], \
                            color="green", linestyle="dashed");
            elif node["xAttrOpt"] == "Y":
                yMin = region[i]["yLower"];
                yMax = region[i]["yUpper"];
                plt.axhline(y=node["xLeftOpt"], \
                            xmin=region[i]["xLower"], xmax=region[i]["xUpper"], \
                            color="green", linestyle="dashed");
        else:
            xLower = region[i]["xLower"];
            xUpper = region[i]["xUpper"];
            yLower = region[i]["yLower"];
            yUpper = region[i]["yUpper"];
            bkColor = "white";
            df = node["dataframe"];
            dataY = listCol(df.values.tolist(), 2);
            prediction = 1.0*sum(dataY)/len(dataY);
            if prediction <= 0.5: bkColor = "red";
            else:                 bkColor = "blue";
            rect=plt.Rectangle((xLower*xRange, yLower*yRange),\
                               (xUpper-xLower)*xRange, (yUpper-yLower)*yRange,\
                               color=bkColor, alpha=0.2);
            ax.add_patch(rect);
    fileNameFig = fileNameTrain.replace("Train.txt", ".png");
    plt.savefig(fileNameFig);
    print("Creating the following file:");
    print(fileNameFig);



###################################################################################
MAXDEPTH = 10;
MINLEAFSIZE = 2;        #make it larger for regression
YVALNAME = "weight";

def biPartVals(xVals):
    xTypeN = len(xVals);    
    xGroups = [];
    for i in range(1, pow(2, xTypeN-1)):
        xGroup = [];
        binaryLabel = bin(i)[2:];
        for j in range(xTypeN - len(binaryLabel)):
            binaryLabel = "0" + binaryLabel;
        for j in range(xTypeN):
            if(binaryLabel[j] == "1"):
                xGroup.append(xVals[j]);
        xGroups.append(xGroup);
    return xGroups;
def dataSelect(data, colIdx, selData, reverse=False):
    dataOutput = [];
    for dataPoint in data:
        if reverse == False:
            if dataPoint[colIdx] in selData:
                dataOutput.append(dataPoint);
        else:
            if dataPoint[colIdx] not in selData:
                dataOutput.append(dataPoint);
    return dataOutput;
def listCol(data, colIdx):
    return [row[colIdx] for row in data];
###################################################################################
def descXGini(data, xColIdx, yColIdx):
    xLeftOpt, giniOpt = None, 999;
    totRowN = len(data);
    if len(list(set(listCol(data, yColIdx)))) == 1:
        return "None", -1;
    for xLefts in biPartVals(list(set(listCol(data, xColIdx)))):
        selDataL = dataSelect(data, xColIdx, xLefts);
        selRowNL = max(1.0, len(selDataL));
        giniL = 1.0;
        for yVal in list(set(listCol(selDataL, yColIdx))):
            giniL -= pow(listCol(selDataL, yColIdx).count(yVal)/selRowNL, 2);
        selDataR = dataSelect(data, xColIdx, xLefts, reverse=True);
        selRowNR = max(1.0, len(selDataR));
        giniR = 1.0;
        for yVal in list(set(listCol(selDataR, yColIdx))):
            giniR -= pow(listCol(selDataR, yColIdx).count(yVal)/selRowNR, 2);
        gini = (selRowNL/totRowN)*giniL + (selRowNR/totRowN)*giniR;
        if gini < giniOpt:
            xLeftOpt = np.copy(xLefts);
            giniOpt = 1.0*gini;
    return xLeftOpt, giniOpt;
def contXGini(data, xColIdx, yColIdx):
    xLeftOpt, giniOpt = 0, 999;
    totRowN = len(data);
    if len(list(set(listCol(data, yColIdx)))) == 1:
        return pow(10, 9), -1;
    for xLeft in list(set(listCol(data, xColIdx))):
        selDataL = [row for row in data if row[xColIdx] <= xLeft];
        selRowNL = max(1.0, len(selDataL));
        giniL = 1.0;
        for yVal in list(set(listCol(selDataL, yColIdx))):
            giniL -= pow(listCol(selDataL, yColIdx).count(yVal)/selRowNL, 2);
        selDataR = [row for row in data if row[xColIdx] > xLeft];
        selRowNR = max(1.0, len(selDataR));
        giniR = 1.0;
        for yVal in list(set(listCol(selDataR, yColIdx))):
            giniR -= pow(listCol(selDataR, yColIdx).count(yVal)/selRowNR, 2);
        gini = (selRowNL/totRowN)*giniL + (selRowNR/totRowN)*giniR;
        if gini < giniOpt:
            xLeftOpt = 1.0*xLeft;
            giniOpt = 1.0*gini;
    return xLeftOpt, giniOpt;
def descXTotSqrRes(data, xColIdx, yColIdx):
    xLeftOpt, totSqrResOpt = None, 999;
    totRowN = len(data);
    if len(list(set(listCol(data, yColIdx)))) == 1:
        return "None", -1;
    for xLeft in list(set(listCol(data, xColIdx))):
        totSqrRes = 0;
        selDataL = [row for row in data if row[xColIdx] <= xLeft];
        meanL = 1.0*sum(listCol(selDataL, yColIdx))/max(1.0, len(selDataL));
        for yVal in listCol(selDataL, yColIdx):
            totSqrRes += pow(yVal - meanL, 2);
        selDataR = [row for row in data if row[xColIdx] > xLeft];
        meanR = 1.0*sum(listCol(selDataR, yColIdx))/max(1.0, len(selDataR));
        for yVal in listCol(selDataR, yColIdx):
            totSqrRes += pow(yVal - meanR, 2);
        if totSqrRes < totSqrResOpt:
            xLeftOpt = 1.0*xLeft;
            totSqrResOpt = 1.0*totSqrRes;
    return xLeftOpt, totSqrResOpt;
def contXTotSqrRes(data, xColIdx, yColIdx):
    xLeftOpt, totSqrResOpt = 0, pow(10, 24);
    totRowN = len(data);
    if len(list(set(listCol(data, yColIdx)))) == 1:
        return pow(10, 24), -1;
    for xLeft in list(set(listCol(data, xColIdx))):
        totSqrRes = 0;
        selDataL = [row for row in data if row[xColIdx] <= xLeft];
        meanL = 1.0*sum(listCol(selDataL, yColIdx))/max(1.0, len(selDataL));
        for yVal in listCol(selDataL, yColIdx):
            totSqrRes += pow(yVal - meanL, 2);
        selDataR = [row for row in data if row[xColIdx] > xLeft];
        meanR = 1.0*sum(listCol(selDataR, yColIdx))/max(1.0, len(selDataR));
        for yVal in listCol(selDataR, yColIdx):
            totSqrRes += pow(yVal - meanR, 2);
        if totSqrRes < totSqrResOpt:
            xLeftOpt = 1.0*xLeft;
            totSqrResOpt = 1.0*totSqrRes;
    return xLeftOpt, totSqrResOpt;
def getScore(dataframe, lastAttr, yAttrName):
    columnNames = dataframe.columns.tolist();
    data = dataframe.values.tolist();
    if len(data) == 0:
        print("ERROR: getScore(): empty data.");
        exit(1); 
    yColIdx = columnNames.index(yAttrName);
    xAttrOpt, xLeftOpt, scoreOpt = "", None, 999;
    for i, columnName in enumerate(columnNames):
        if (i != yColIdx) and (columnName != lastAttr):
            if "str" in str(type(data[0][i])):
                if "str" in str(type(data[0][yColIdx])):
                    xLeft, score = descXGini(data, i, yColIdx);
                elif "float" in str(type(data[0][yColIdx])):
                    xLeft, score = descXTotSqrRes(data, i, yColIdx);
                else:
                    print("ERROR: getScore(): unknown type for " + \
                          data[0][yColIdx] + ".");
                    exit(1);
                if score < scoreOpt:
                    xAttrOpt = columnName;
                    xLeftOpt = copy.copy(xLeft);
                    scoreOpt = 1.0*score;
            elif "float" in str(type(data[0][i])):
                if "str" in str(type(data[0][yColIdx])):
                    xLeft, score = contXGini(data, i, yColIdx);
                elif "float" in str(type(data[0][yColIdx])):
                    xLeft, score = contXTotSqrRes(data, i, yColIdx);
                else:
                    print("ERROR: getScore(): unknown type for " + \
                          data[0][yColIdx] + ".");
                if score < scoreOpt:
                    xAttrOpt = columnName;
                    xLeftOpt = 1.0*xLeft;
                    scoreOpt = 1.0*score;
            else:
                print("ERROR: getScore(): unknown type for " + data[0][i] + ".");
                exit(1);
    return xAttrOpt, xLeftOpt, scoreOpt;
###################################################################################
def withoutKeys(node, keys):
    return {x: node[x] for x in node if x not in keys};
def buildTree(dataframe, yAttrName):
    columnNames = dataframe.columns.tolist();
    if yAttrName not in columnNames:
        print("ERROR: buildTree(): column name " + yAttrName + " not found.");
        exit(0);
    tree = [];
    node = genNode(0, 0, dataframe, -1, "root", "", yAttrName);
    tree.append(node);
    splitNode(tree, node);
    return tree;
def genNode(index, depth, dataframe, parentIdx, nodeDir, xAttrLast, yAttrName):
    xAttrOpt, xLeftOpt, scoreOpt = getScore(dataframe, xAttrLast, yAttrName);
    return {"index": index, "depth": depth, "isLeaf": "No", 
            "parent": parentIdx, "nodeDir": nodeDir, "yAttrName": yAttrName,\
            "xAttrOpt": xAttrOpt, "xLeftOpt": xLeftOpt, "scoreOpt": scoreOpt,\
            "dataCount": dataframe.count()[-1], "dataframe": dataframe};
def splitNode(tree, node):
    index = node["index"];
    depth = node["depth"];
    if depth > MAXDEPTH:
        node["isLeaf"] = "Exceed max depth"; 
        return;
    yAttrName = node["yAttrName"];
    score = node["scoreOpt"];
    xAttrOpt = node["xAttrOpt"];
    xLeftOpt = node["xLeftOpt"];
    dataframe = node["dataframe"];
    dfLeft, dfRight = None, None;
    xLeftOptType = str(type(xLeftOpt));
    if "list" in xLeftOptType:
        xLeftOptType = str(type(xLeftOpt[0]));
    if "str" in xLeftOptType:
        dfLeft  = dataframe.loc[dataframe[xAttrOpt].isin(xLeftOpt)];
        dfRight = dataframe.loc[~dataframe[xAttrOpt].isin(xLeftOpt)];
    elif "float" in xLeftOptType:
        dfLeft  = dataframe.loc[dataframe[xAttrOpt] <= xLeftOpt];
        dfRight = dataframe.loc[dataframe[xAttrOpt] > xLeftOpt];
    else:
        print("ERROR: splitNode(): unknown type for " + xLeftOpt + ".");
        exit(1);
    if (dfLeft.count()[-1] < MINLEAFSIZE) or (dfRight.count()[-1] < MINLEAFSIZE):
        node["isLeaf"] = "Under min leaf size: ";
        node["isLeaf"] += str(dfLeft.count()[-1]) + "|" + str(dfRight.count()[-1]);
        return;
    del(node["dataframe"]);
    if dfLeft.count()[-1] != 0:
        indexLeft = len(tree);
        nodeLeft = genNode(indexLeft, depth+1, dfLeft, \
                           index, "left", xAttrOpt, yAttrName);
        tree.append(nodeLeft);
        if nodeLeft["scoreOpt"] == -1:
            tree[indexLeft]["isLeaf"] = "Already pure";
        elif nodeLeft["scoreOpt"] < score:
            splitNode(tree, tree[indexLeft]);
        else:
            tree[indexLeft]["isLeaf"] = "Insufficient score"; 
    if dfRight.count()[-1] != 0:
        indexRight = len(tree);
        nodeRight = genNode(indexRight, depth+1, dfRight, \
                            index, "right", xAttrOpt, yAttrName);
        tree.append(nodeRight);
        if nodeRight["scoreOpt"] == -1:
            tree[indexRight]["isLeaf"] = "Already pure";
        elif nodeRight["scoreOpt"] < score:
            splitNode(tree, tree[indexRight]);
        else:
            tree[indexRight]["isLeaf"] = "Insufficient score";
def printTree(tree):
    print("yAttrName = " + tree[0]["yAttrName"]);
    for node in tree:
        print("--------------------------------------------------------------");
        print(withoutKeys(node, ["yAttrName"]));
###################################################################################
def getPredictedY(tree, columnNames, dataPoint, verbosity):
    if len(dataPoint.shape) != 1:
        print("ERROR: predYSingle(): Need a single data point.");
        exit(0);
    node = tree[0];
    yAttrName = node["yAttrName"];
    yColIdx = columnNames.index(yAttrName);
    xLeftOpt = node["xLeftOpt"];
    if verbosity > 0:
        print(dataPoint);
        print(node)
    while node["isLeaf"] == "No":
        nodeIdx = node["index"];
        xColIdx = columnNames.index(node["xAttrOpt"]); 
        selCond = None;
        xLeftOptType = str(type(xLeftOpt));
        if "list" in xLeftOptType:
            xLeftOptType = str(type(xLeftOpt[0])); 
        if "str" in xLeftOptType:
            selCond = (dataPoint[xColIdx] in xLeftOpt);
        elif "float" in xLeftOptType:
            selCond = (dataPoint[xColIdx] <= xLeftOpt);
        else:
            print("ERROR: getPredictedY(): unknown type for " + xLeftOpt + ".");
            exit(1);
        if selCond == True:
            subTree = [n for n in tree if n["parent"] == nodeIdx and \
                                          n["nodeDir"] == "left"];
        else:
            subTree = [n for n in tree if n["parent"] == nodeIdx and \
                                          n["nodeDir"] == "right"];
        if len(subTree) != 1:
            print("ERROR: predYSingle(): Wrong tree configuration 1.");
            exit(1);
        node = subTree[0];
        if verbosity > 0:
            print(node);
    df = node["dataframe"];
    try:
        df = node["dataframe"];
        dataY = listCol(df.values.tolist(), yColIdx);
        prediction = None;
        if "str" in str(type(dataY[0])):
            prediction = df.mode(axis=0)[yAttrName][0];
        elif "float" in str(type(dataY[0])):
            prediction = 1.0*sum(dataY)/len(dataY);
        if verbosity > 0:
            print("prediction: ", prediction);
        return prediction;
    except:
        print("ERROR: predYSingle(): Wrong tree configuration 2.");
        exit(1);
def predictY(tree, dataframe, verbosity=0):
    columnNames = dataframe.columns.tolist();
    yAttrName = tree[0]["yAttrName"];
    yColIdx = columnNames.index(yAttrName);
    data = dataframe.to_numpy();
    for i, dataPoint in enumerate(data):
        predictedY = getPredictedY(tree, columnNames, dataPoint, verbosity);
        data[i][yColIdx] = predictedY;
    return data;
###################################################################################
if __name__ == "__main__": main();












 
