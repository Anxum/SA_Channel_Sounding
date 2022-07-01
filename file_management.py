import os
import pandas as pd
import numpy as np

def read_meta_data(filename):
    str = filename.split('_')
    date_ = str[1]
    time_ = str[2]
    fc_ = int( str[3][0:str[3].find('MHz')] ) * 10**6
    fs_ = int( str[4][0:str[4].find('MSps')] ) * 10**6
    batchsize_ = int( str[5][0:str[5].find('S')] )
    capture_interval_ = int( str[6][0:str[6].find('ms')] ) * 10**-3
    return fc_, fs_, batchsize_, capture_interval_, date_, time_

def choose_measurement():
    df = pd.read_csv("../list_of_measurements.csv")
    path = []
    while True:
        name = input("Please enter the name of the file you want to evaluate: ")
        path = np.array(df[df["Name"] == name]["Path to file"])
        if path.size == 0:
            print("The name you entered does not exist!")
            print("Possible names are:")
            print(df["Name"])
            path = ""
        else:
            break
    split_idx = path[0].rfind('/')
    folder = path[0][:split_idx+1]
    file = path[0][split_idx+1:]
    return folder, file


def register_measurements():
    ignore_list = []
    try:
        f = open("../measurement_registration_ignore.txt", "r")
        ignore_list = f.read().splitlines()
        f.close()
    except:
        pass
    home = os.path.expanduser("~")
    unresolved_files = []
    for item in os.listdir(home):
        if ".dat" in item and not item in ignore_list:
            unresolved_files.append(item)
    while unresolved_files:
        file = unresolved_files.pop()
        options = {
        "1":"Register and save the measurement",
        "2":"Discard the measurement",
        "3":"Ignore this measurement once",
        "4":"Always ignore this measurement"
        }

        print(f"=====Unregistered measurement found:=====")
        print("")
        print(file)
        print("")
        print("Please choose one of the following options: ")
        for number in options:
            print(f"{number} ---- {options[number]}")
        print("")

        while(True):
            option = input("Please enter an option: ")
            print("")
            if option in options.keys():
                break
            else:
                print("Please enter a number that corresponds with the options above!")


        if option == "1": # Register a file
            df = pd.read_csv("../list_of_measurements.csv", index_col = 0)
            fc, fs, batchsize, capture_interval, date, time = read_meta_data(file)
            description = input("Please enter a brief description of the measurement: ")
            name = ""
            while True:
                name = input("Please enter a name for the measurement: ")
                if not name in df["Name"].values:
                    break
                else:
                    print("Name does already exist! Please choose another one.")

            df.loc[len(df.index)] = {
            "Name": name,
            "Date": date,
            "Time": time,
            "fc": fc,
            "fs": fs,
            "Batchsize": batchsize,
            "Capture interval": capture_interval,
            "Path to file": f"../Messungen/{file}",
            "Comment": description}

            os.rename(f"{home}/{file}", f"../Messungen/{file}")
            df.to_csv("../list_of_measurements.csv")
            print(f"Measurement was saved to /Messungen/{file} with the name: {name}")

        if option == "2": # Delete a file

            print("")
            print("Are you sure you want to delete this measurement?")
            print(file)
            print("")
            if input(" y/n  - ") in ["y", "Y"]:
                os.remove(f"{home}/{file}")
                print("")
                print("File has been removed.")
            else:
                unresolved_files.append(file)

        if option == "3": # Ignore the file for now
            pass
        if option == "4": # Always ignore the file
            ignore_list.append(file)
            f = open("../measurement_registration_ignore.txt", "a")
            f.write(f"{file}\n")
            f.close()
