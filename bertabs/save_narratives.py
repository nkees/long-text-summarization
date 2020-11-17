with open("train_documents.txt", "r") as file:
    narratives = file.read()
    narratives = narratives.split("\n")
    n = 1
    for i in narratives[0:101]:
        file_name = "data/narrative_{}.txt".format(n)
        # print("NARRATIVE " + str(n))
        # print(i)
        n += 1
        # command = input()
        # if command == "":
        #    continue
        with open(file_name, "w") as document:
            document.write(i)

with open("train_labels.txt", "r") as file:
    labels = file.read()
    labels = labels.split("\n")
    n = 1
    for i in labels[0:101]:
        file_name = "gold/label_{}.txt".format(n)
        # print("NARRATIVE " + str(n))
        # print(i)
        n += 1
        # command = input()
        # if command == "":
        #    continue
        # else:
        #     break
        with open(file_name, "w") as document:
            document.write(i)
