import numpy as np
import pandas as pd
import tflearn

trained_data = pd.read_json("Nom_output_orig", lines = True)

trained_data = pd.DataFrame(trained_data)

print(trained_data)
possible_response = ["y","yes","yap","yeah",","]
while True:
    user_input = input("Type description to get HS code and best match from my database : ")

    top_five = trained_data[trained_data["Description"].str.contains(user_input, case = False, regex = True)].sort_values("Description", ascending=False).head(5)
    print(top_five)

    response = input("type --- 'exit' ---- to quit the script or anything else to continue: ")
    if response.lower() == "exit":
        break
print("So, we are done here :)")


"""net = tflearn.input_data(shape=[None, len(trained_data[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)"""