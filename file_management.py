import os
import pandas as pd
import numpy as np

def read_meta_data(filename):
    str = filename.split('_')
    date_ = str[1]
    time_ = str[2]
    fc_ = int( str[3][0:str[3].find('MHz')] ) * 1e6
    fs_ = int( str[4][0:str[4].find('MSps')] ) * 1e6
    batchsize_ = int( str[5][0:str[5].find('S')] )
    capture_interval_ = int( str[6][0:str[6].find('ms')] ) * 1e-3
    return fc_, fs_, batchsize_, capture_interval_, date_, time_



def choose_measurement():
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv("../list_of_measurements.csv", index_col = 0)
    path = []
    while True:
        print("===========================================================================")
        print("")
        print("Options:")
        print("1 ---- Show all names of the measurements with the corresponding descriptions")
        print("2 ---- Remove a measurement from the registration")
        print("3 ---- End the program")
        option = input("Or enter the name of the file you want to evaluate (Press Enter to evaluate the last measurement):  ")
        print("")
        if option == "1": # Show all measurements
            print("")
            print(df[["Name", "Description"]])
            continue

        if option == "2": # Remove measurement from registration
            remove = input("Please enter the name of the measurement you want to remove: ")
            print("")
            rem_idx = df.index[df["Name"] == remove].tolist()
            if len(rem_idx) == 0:
                print("The name you entered does not exist!")
                print("Possible names are:")
                print(df["Name"])
                continue
            print("_____________________________________________________________")
            print(df.iloc[rem_idx[0]])
            print("_____________________________________________________________")
            print("")
            print("Are you sure you want to remove this measurement:")
            confirmation = input( "y/n - : ")
            if confirmation in ["y", "yes", "Y", "YES"]:
                rem_path = df.iloc[rem_idx[0]]["Path to file"]
                split_idx = rem_path.rfind('/')
                file = rem_path[split_idx+1:]
                os.remove(rem_path)
                df.drop([rem_idx[0]], inplace = True)
                df.reset_index()
                df.to_csv("../list_of_measurements.csv")
                print("")
                print(f"Measurement {remove} has been removed from registration")
            continue

        if option == "3": # Exit Program
            return "","",""

        name = option                   #A Name has been entered
        path = np.array(df[df["Name"] == name]["Path to impulse response"])

        if name == "":
            path = np.array([df["Path to impulse response"].iloc[-1]]) # Choose the latest measurement
            name = df["Name"].iloc[-1]

        if path[0] == "-":
            path = np.array(df[df["Name"] == name]["Path to file"])

        if path.size == 0:
            print("The name you entered does not exist!")
            print("Possible names are:")
            print(df["Name"])
            path = ""
        else:
            break
    split_idx = path[0].rfind('/')
    folder = path[0][:split_idx]
    file = path[0][split_idx+1:]
    return folder, file, name

def save_impulse_response(h, date, time, fc, fs, batchsize, capture_interval, name_of_measurement, path_to_raw_measurement):
    df = pd.read_csv("../list_of_measurements.csv", index_col = 0)
    measurement_idx = df.index[df["Name"] == name_of_measurement].tolist()[0]
    if df["Path to impulse response"].iloc[measurement_idx]  == "-":
        filename = f"{name_of_measurement}_{date}_{time}_{int(fc * 1e-6)}MHz_{int(fs *1e-6)}MSps_{batchsize}S_{int(capture_interval * 1e3)}ms"
        folder = "../impulse_responses"
        df.loc[measurement_idx, "Path to impulse response"] = f"{folder}/{filename}.npy"
        np.save(f"{folder}/{filename}", h)
        df.to_csv("../list_of_measurements.csv")


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
        "4":"Always ignore this measurement",
        "5":"Skip this step"
        }

        print(f"======================Unregistered measurement found:======================")
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
                if name in ["1", "2", "3", "4", "5", "6", "7" ,"8", "9", "0"]:
                    print("Please do not choose a single number as a name for the measurement!")
                elif not name in df["Name"].values:
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
            "Path to file": f"../raw_data/{file}",
            "Path to impulse response": "-",
            "Description": description}

            os.rename(f"{home}/{file}", f"../raw_data/{file}")
            df.to_csv("../list_of_measurements.csv")
            print(f"Measurement was saved to /raw_data/{file} with the name: {name}")

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
        if option == "5": # End the program
            break
