import numpy as np
import pandas as pd
import tflearn

trained_data = pd.read_json("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", lines = True)

trained_data = pd.DataFrame(trained_data)

print(trained_data)
possible_response = ["y","yes","yap","yeah",","]
while True:
    user_input = input("Type description to get HS code and best match from my database : ")

    top_five = trained_data[trained_data["Description"].str.contains(user_input, case = False, regex = True)].sort_values("Description", ascending=False).head(5)
    print(top_five)

    response = input("Do you want to run the script again? y/n: ")
    if response.lower() not in possible_response:
        break
print("So, we are done here :)")


"""net = tflearn.input_data(shape=[None, len(trained_data[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)"""