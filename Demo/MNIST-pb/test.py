import xml.dom.minidom

dom=xml.dom.minidom.parse("conf.xml")
root=dom.documentElement
DATA_DIR=root.getElementsByTagName('DATA_DIR')[0].firstChild.data
MODEL_DIR=root.getElementsByTagName('MODEL_DIR')[0].firstChild.data
PB_FILENAME=root.getElementsByTagName('PB_FILENAME')[0].firstChild.data
LOG_FILENAME=root.getElementsByTagName('LOG_FILENAME')[0].firstChild.data
BATCH_SIZE=root.getElementsByTagName('BATCH_SIZE')[0].firstChild.data
TRAIN_TIMES=root.getElementsByTagName('TRAIN_TIMES')[0].firstChild.data
KEEP_PROB=root.getElementsByTagName('KEEP_PROB')[0].firstChild.data
W_CONV1_NAME=root.getElementsByTagName('W_CONV1_NAME')[0].firstChild.data
B_CONV1_NAME=root.getElementsByTagName('B_CONV1_NAME')[0].firstChild.data
W_CONV2_NAME=root.getElementsByTagName('W_CONV2_NAME')[0].firstChild.data
B_CONV2_NAME=root.getElementsByTagName('B_CONV2_NAME')[0].firstChild.data
W_CONN3_NAME=root.getElementsByTagName('W_CONN3_NAME')[0].firstChild.data
B_CONN3_NAME=root.getElementsByTagName('B_CONN3_NAME')[0].firstChild.data
W_FINAL5_NAME=root.getElementsByTagName('W_FINAL5_NAME')[0].firstChild.data
B_FINAL5_NAME=root.getElementsByTagName('B_FINAL5_NAME')[0].firstChild.data