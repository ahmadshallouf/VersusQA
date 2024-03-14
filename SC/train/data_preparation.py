import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
train_df = pd.read_csv("./data/comparg_train.tsv", sep="\t")
test_df = pd.read_csv("./data/comparg_test.tsv", sep="\t")

# only keep the columns answer, answer_stance, object_1 and object_2
train_df = train_df[["object_0", "object_1", "answer", "labels"]]
test_df = test_df[["object_0", "object_1", "answer", "labels"]]

# shuffle the dataset
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# rename the column most_frequent_label to y
train_df = train_df.rename(columns={"labels": "y"})
test_df = test_df.rename(columns={"labels": "y"})

# concatenate object_a and object_b into one column
train_df["objects"] = train_df["object_0"] + " [SEP] " + train_df["object_1"]
test_df["objects"] = test_df["object_0"] + " [SEP] " + test_df["object_1"]

# drop object_a and object_b columns
train_df = train_df.drop(["object_0", "object_1"], axis=1)
test_df = test_df.drop(["object_0", "object_1"], axis=1)

# concatenate objects and sentence into one column seperated by a comma
train_df["answer"] = train_df["objects"] + " [SEP] " + train_df["answer"]
test_df["answer"] = test_df["objects"] + " [SEP] " + test_df["answer"]

# drop the objects column
train_df = train_df.drop(["objects"], axis=1)
test_df = test_df.drop(["objects"], axis=1)

# rename the column sentence to x
train_df = train_df.rename(columns={"answer": "x"})
test_df = test_df.rename(columns={"answer": "x"})

# rearange columns so that column x is the first one
train_df = train_df[["x", "y"]]
test_df = test_df[["x", "y"]]

# save the new dataset to a csv file with same name as the original file but in folder data
# original file name is in variable file
train, val = train_test_split(train_df, test_size=0.1, random_state=0)
train.to_csv("./data/train2.csv", index=False, sep="\t")
val.to_csv("./data/val2.csv", index=False, sep="\t")
test_df.to_csv("./data/test2.csv", index=False, sep="\t")
